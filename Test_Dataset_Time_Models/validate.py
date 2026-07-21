"""Validate outdoor time models against the men's scraped test dataset.

Uses formulas from models/steeplechase band reports.
Evaluates:
  1) Season-PB → season-PB (same setup the models were fit on)
  2) Chronological: earlier from-event mark → later to-event mark

Reports absolute/relative errors, tolerance hit rates, and comparison to
each model's reported training CV median |error|.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models" / "steeplechase"
TEST_CSV = ROOT / "output" / "men_test_dataset_results.csv"
OUT_DIR = ROOT / "output" / "model_validation"

BAND_FILES = {
    "750-950": "new_format_feature_important_time_models_band_750_950_with_steeple.txt",
    "800-1000": "new_format_feature_important_time_models_band_800_1000_with_steeple.txt",
    "850-1050": "new_format_feature_important_time_models_band_850_1050_with_steeple.txt",
}

# Model file event name → test-dataset event name
MODEL_TO_TEST_EVENT = {
    "100m": "100m",
    "200m": "200m",
    "400m": "400m",
    "800m": "800m",
    "1500m": "1500m",
    "3000m": "3000m",
    "3000m Steeplechase": "3000m SC",
    "5000m": "5000m",
}
TEST_TO_MODEL_EVENT = {v: k for k, v in MODEL_TO_TEST_EVENT.items()}

TRACK_EVENTS = set(MODEL_TO_TEST_EVENT.values())

# Starter absolute tolerances (seconds) from preprocessing design
TOLERANCE = {
    "100m": 0.15,
    "200m": 0.30,
    "400m": 0.80,
    "800m": 1.50,
    "1500m": 3.00,
    "3000m": 6.00,
    "3000m SC": 8.00,
    "5000m": 10.00,
}


@dataclass
class Formula:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    applicable: bool = True  # False for knn without neighbor store

    def predict(self, x: float) -> float | None:
        if not self.applicable or x is None or not math.isfinite(x) or x <= 0:
            return None
        k = self.kind
        p = self.params
        try:
            if k in ("linear", "robust_trimmed_linear", "linear_ols"):
                return p["intercept"] + p["slope"] * x
            if k == "median_ratio":
                return p["ratio"] * x
            if k == "log_linear":
                return math.exp(p["intercept"] + p["slope"] * math.log(x))
            if k == "ratio_linear":
                return x * (p["a"] + p["b"] * x)
            if k == "quadratic":
                return p["a"] + p["b"] * x + p["c"] * x * x
            if k == "binned_ratio":
                edges = p["edges"]
                ratios = p["ratios"]
                if not edges:
                    return None
                # nearest edge by |x - edge|
                i = min(range(len(edges)), key=lambda j: abs(x - edges[j]))
                return x * ratios[i]
            if k == "knn_fallback_ratio":
                return p["ratio"] * x
            if k == "knn_median":
                return None
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
        return None


@dataclass
class PairModel:
    band: str
    gender: str
    event_group: str
    from_event: str  # model naming
    to_event: str
    n_train: int
    r: float | None
    pooled_model_name: str
    pooled: Formula
    pooled_cv: float | None
    pooled_full_medae: float | None
    median_ratio: float | None
    recommended_strategy: str | None
    recommended_cv: float | None
    cohorts: dict[str, Formula]  # label -> formula for recommended strategy
    strategy_name: str | None = None


def _parse_float(s: str) -> float:
    return float(s.replace("−", "-").replace("–", "-").strip())


def parse_formula_string(expr: str, from_ev: str, to_ev: str) -> Formula:
    """Parse a single formula expression into a Formula object."""
    raw = expr.strip()
    # Normalize operators; strip spaces from expr AND event names for matching
    s = (
        raw.replace("−", "-")
        .replace("–", "-")
        .replace("×", "*")
        .replace("²", "^2")
        .replace(" ", "")
    )
    from_ev_n = from_ev.replace(" ", "")
    to_ev_n = to_ev.replace(" ", "")

    if (
        "medianofk=" in s.lower()
        or "≈median" in s.lower()
        or ("neighbors" in s.lower() and "median" in s.lower())
    ):
        return Formula(kind="knn_median", raw=raw, applicable=False)
    if "ratio_bin" in s:
        return Formula(kind="binned_ratio", params={"edges": [], "ratios": []}, raw=raw)

    m = re.search(
        rf"^{re.escape(to_ev_n)}=exp\(([+-]?\d+(?:\.\d+)?)\+([+-]?\d+(?:\.\d+)?)\*log\({re.escape(from_ev_n)}\)\)$",
        s,
    )
    if m:
        return Formula(
            "log_linear",
            {"intercept": float(m.group(1)), "slope": float(m.group(2))},
            raw,
        )

    m = re.search(
        rf"^{re.escape(to_ev_n)}={re.escape(from_ev_n)}\*\(([+-]?\d+(?:\.\d+)?)([+-])(\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}\)$",
        s,
    )
    if m:
        b = float(m.group(3))
        if m.group(2) == "-":
            b = -b
        return Formula("ratio_linear", {"a": float(m.group(1)), "b": b}, raw)

    m = re.search(
        rf"^{re.escape(to_ev_n)}=([+-]?\d+(?:\.\d+)?)\+([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}([+-])(\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}\^2$",
        s,
    )
    if m:
        c = float(m.group(4))
        if m.group(3) == "-":
            c = -c
        return Formula(
            "quadratic",
            {"a": float(m.group(1)), "b": float(m.group(2)), "c": c},
            raw,
        )
    m = re.search(
        rf"^{re.escape(to_ev_n)}=([+-]?\d+(?:\.\d+)?)\+([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}\+([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}\^2$",
        s,
    )
    if m:
        return Formula(
            "quadratic",
            {"a": float(m.group(1)), "b": float(m.group(2)), "c": float(m.group(3))},
            raw,
        )

    m = re.search(
        rf"^{re.escape(to_ev_n)}=([+-]?\d+(?:\.\d+)?)\+([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}$",
        s,
    )
    if m:
        return Formula(
            "linear",
            {"intercept": float(m.group(1)), "slope": float(m.group(2))},
            raw,
        )
    m = re.search(
        rf"^{re.escape(to_ev_n)}=([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}-(\d+(?:\.\d+)?)$",
        s,
    )
    if m:
        return Formula(
            "linear",
            {"intercept": -float(m.group(2)), "slope": float(m.group(1))},
            raw,
        )
    m = re.search(
        rf"^{re.escape(to_ev_n)}=([+-]?\d+(?:\.\d+)?)\*{re.escape(from_ev_n)}$",
        s,
    )
    if m:
        return Formula("median_ratio", {"ratio": float(m.group(1))}, raw)

    return Formula(kind="unparsed", raw=raw, applicable=False)


def parse_band_file(path: Path, band: str) -> list[PairModel]:
    text = path.read_text()
    lines = text.splitlines()
    pairs: list[PairModel] = []
    gender = None
    event_group = None
    i = 0
    pair_header = re.compile(
        r"^(?P<frm>.+?) -> (?P<to>.+?)\s+\(n=(?P<n>\d+) athlete-seasons(?:,\s*r=(?P<r>-?\d+(?:\.\d+)?))?\)\s*$"
    )

    while i < len(lines):
        line = lines[i]
        if line.startswith("Sprints — ") or line.startswith("Distance — "):
            parts = line.split("—")
            event_group = parts[0].strip()
            gender = parts[1].strip()
            i += 1
            continue

        m = pair_header.match(line.strip())
        if not m or gender != "Men":
            i += 1
            continue

        from_ev = m.group("frm").strip()
        to_ev = m.group("to").strip()
        n_train = int(m.group("n"))
        r = float(m.group("r")) if m.group("r") else None

        pooled_name = None
        pooled_formula_line = None
        pooled_cv = None
        pooled_full = None
        median_ratio = None
        recommended = None
        recommended_cv = None
        bin_edges: list[float] = []
        bin_ratios: list[float] = []
        cohorts: dict[str, Formula] = {}
        strategy_name = None

        i += 1
        while i < len(lines):
            L = lines[i]
            if pair_header.match(L.strip()) or L.startswith("Sprints — ") or L.startswith("Distance — ") or L.startswith("Coverage"):
                break
            if L.startswith("Women —") or L.startswith("Sprints — Women") or L.startswith("Distance — Women"):
                break

            if L.strip().startswith("Pooled band model:"):
                pooled_name = L.split(":", 1)[1].strip()
            elif L.strip().startswith("Formula:"):
                pooled_formula_line = L.split(":", 1)[1].strip()
            elif "nearest " in L and "→" in L:
                bm = re.search(
                    rf"nearest\s+{re.escape(from_ev)}\s+(-?\d+(?:\.\d+)?)s\s+→\s+(-?\d+(?:\.\d+)?)",
                    L,
                )
                if bm:
                    bin_edges.append(float(bm.group(1)))
                    bin_ratios.append(float(bm.group(2)))
            elif "CV median |error| (pooled routing):" in L:
                pooled_cv = float(re.search(r"([\d.]+)s", L).group(1))
            elif "Full-sample median |error|:" in L:
                pooled_full = float(re.search(r"([\d.]+)s", L).group(1))
            elif L.strip().startswith(f"Ratio {to_ev}/{from_ev}:"):
                rm = re.search(r"median\s+([\d.]+)", L)
                if rm:
                    median_ratio = float(rm.group(1))
            elif "Recommended route:" in L:
                rm = re.search(r"Recommended route:\s*(\S+)\s*\(CV\s*([\d.]+)s", L)
                if rm:
                    recommended = rm.group(1)
                    recommended_cv = float(rm.group(2))
            elif L.strip().startswith("How ") and " alters the formula:" in L:
                strategy_name = L.strip()[4:].split(" alters")[0].strip()
            elif L.strip().startswith("[") and "] n=" in L and ":" in L:
                # [label] n=N: formula (CV xs)
                cm = re.match(
                    r"\s*\[(?P<label>[^\]]+)\]\s+n=(?P<n>\d+):\s+(?P<form>.+?)(?:\s+\(CV\s+(?P<cv>[\d.]+)s\))?\s*$",
                    L,
                )
                if cm:
                    label = cm.group("label").strip()
                    form = cm.group("form").strip()
                    # strip trailing CV if still attached
                    form = re.sub(r"\s+\(CV\s+[\d.]+s\)\s*$", "", form)
                    fobj = parse_formula_string(form, from_ev, to_ev)
                    # peek following nearest lines for binned
                    j = i + 1
                    edges, ratios = [], []
                    while j < len(lines) and "nearest " in lines[j] and "→" in lines[j]:
                        bm = re.search(
                            rf"nearest\s+{re.escape(from_ev)}\s+(-?\d+(?:\.\d+)?)s\s+→\s+(-?\d+(?:\.\d+)?)",
                            lines[j],
                        )
                        if bm:
                            edges.append(float(bm.group(1)))
                            ratios.append(float(bm.group(2)))
                        j += 1
                    if fobj.kind == "binned_ratio":
                        fobj.params = {"edges": edges, "ratios": ratios}
                    cohorts[label] = fobj
            i += 1

        if not pooled_formula_line or not pooled_name:
            continue

        pooled = parse_formula_string(pooled_formula_line, from_ev, to_ev)
        if pooled.kind == "binned_ratio":
            pooled.params = {"edges": bin_edges, "ratios": bin_ratios}
        # knn fallback to reported median ratio when available
        if pooled.kind == "knn_median" and median_ratio is not None:
            pooled = Formula(
                "knn_fallback_ratio",
                {"ratio": median_ratio},
                raw=f"FALLBACK median_ratio {median_ratio} (knn unavailable)",
                applicable=True,
            )

        pairs.append(
            PairModel(
                band=band,
                gender=gender or "Men",
                event_group=event_group or "",
                from_event=from_ev,
                to_event=to_ev,
                n_train=n_train,
                r=r,
                pooled_model_name=pooled_name,
                pooled=pooled,
                pooled_cv=pooled_cv,
                pooled_full_medae=pooled_full,
                median_ratio=median_ratio,
                recommended_strategy=recommended,
                recommended_cv=recommended_cv,
                cohorts=cohorts,
                strategy_name=strategy_name or recommended,
            )
        )
    return pairs


def load_test_pb(df: pd.DataFrame) -> pd.DataFrame:
    """Season PB = fastest Result_Value per athlete-season-event (track)."""
    track = df[df["Event"].isin(TRACK_EVENTS)].copy()
    track["Competition_Date"] = pd.to_datetime(track["Competition_Date"], errors="coerce")
    pb = (
        track.sort_values(["Result_Value", "Competition_Date"], na_position="last")
        .groupby(["College", "Athlete", "Season_Year", "Event"], as_index=False)
        .first()
    )
    return pb


def athlete_in_band(df: pd.DataFrame, lo: int, hi: int) -> set[tuple]:
    """Athlete-seasons with ≥1 result in WA [lo, hi)."""
    sub = df[(df["World_Athletics_Score_Men"] >= lo) & (df["World_Athletics_Score_Men"] < hi)]
    return set(zip(sub["College"], sub["Athlete"], sub["Season_Year"]))


def cohort_label(strategy: str, row_from: pd.Series, row_to: pd.Series, best_event: str, bal_spec: str) -> str | None:
    if strategy == "bal_spec":
        return bal_spec
    if strategy == "best_event":
        return f"best_{best_event}" if best_event else None
    if strategy == "best_is_from":
        return "best_is_from" if best_event == row_from["Event"] else "best_not_from"
    if strategy == "pair_wa_gap_50":
        gap = abs(float(row_from["World_Athletics_Score_Men"]) - float(row_to["World_Athletics_Score_Men"]))
        return "pair_gap_ge50" if gap >= 50 else "pair_gap_lt50"
    if strategy == "from_stronger_wa":
        if float(row_from["World_Athletics_Score_Men"]) >= float(row_to["World_Athletics_Score_Men"]):
            return "from_stronger_wa"
        return "from_weaker_wa"
    if strategy == "bal_x_best_event":
        be = f"best_{best_event}" if best_event else None
        if not be or not bal_spec:
            return None
        return f"{bal_spec} ∩ {be}"
    return None


def pick_feature_formula(pm: PairModel, label: str | None) -> tuple[Formula, str]:
    """Return (formula, route_used). Fall back to pooled if cohort missing."""
    if label and label in pm.cohorts and pm.cohorts[label].applicable:
        f = pm.cohorts[label]
        if f.kind == "knn_median" and pm.median_ratio is not None:
            f = Formula("knn_fallback_ratio", {"ratio": pm.median_ratio}, applicable=True)
        if f.applicable and f.kind != "unparsed":
            return f, f"feature:{pm.recommended_strategy}:{label}"
    # fallbacks for bal_x_best_event
    if pm.recommended_strategy == "bal_x_best_event" and label:
        # try best_event alone, then bal_spec alone
        parts = label.split(" ∩ ")
        if len(parts) == 2:
            for alt in (parts[1], parts[0]):
                if alt in pm.cohorts and pm.cohorts[alt].applicable:
                    return pm.cohorts[alt], f"feature_fallback:{alt}"
    return pm.pooled, "pooled"


def err_stats(errors: list[float]) -> dict[str, float]:
    if not errors:
        return {}
    a = np.array(errors, dtype=float)
    return {
        "n": int(len(a)),
        "mae": float(np.mean(np.abs(a))),
        "medae": float(np.median(np.abs(a))),
        "rmse": float(np.sqrt(np.mean(a**2))),
        "bias_mean": float(np.mean(a)),  # predicted - actual
        "p90_abs": float(np.percentile(np.abs(a), 90)),
    }


def evaluate(
    models: list[PairModel],
    df: pd.DataFrame,
    pb: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """mode: 'season_pb' or 'chronological'."""
    rows_out: list[dict] = []

    # Precompute group best_event / bal_spec from test features when present
    for pm in models:
        from_test = MODEL_TO_TEST_EVENT.get(pm.from_event)
        to_test = MODEL_TO_TEST_EVENT.get(pm.to_event)
        if not from_test or not to_test:
            continue
        lo, hi = map(int, pm.band.split("-"))
        in_band = athlete_in_band(df, lo, hi)

        if mode == "season_pb":
            from_pb = pb[pb["Event"] == from_test]
            to_pb = pb[pb["Event"] == to_test]
            merged = from_pb.merge(
                to_pb,
                on=["College", "Athlete", "Season_Year"],
                suffixes=("_from", "_to"),
            )
            candidates = []
            for _, r in merged.iterrows():
                key = (r["College"], r["Athlete"], int(r["Season_Year"]))
                if key not in in_band:
                    continue
                candidates.append(r)
        else:
            # chronological: dated rows only (supplementary undated season-PBs excluded)
            track = df[df["Event"].isin([from_test, to_test])].copy()
            if "Source_Role" in track.columns:
                track = track[track["Source_Role"] != "supplementary_season_pb"]
            track["Competition_Date"] = pd.to_datetime(track["Competition_Date"], errors="coerce")
            track = track.dropna(subset=["Competition_Date"])
            candidates = []
            for key, g in track.groupby(["College", "Athlete", "Season_Year"]):
                if key not in in_band:
                    continue
                frm = g[g["Event"] == from_test].sort_values("Competition_Date")
                too = g[g["Event"] == to_test].sort_values("Competition_Date")
                if frm.empty or too.empty:
                    continue
                # earliest from that has a later to; take first such from + first later to
                paired = False
                for _, fr in frm.iterrows():
                    later = too[too["Competition_Date"] > fr["Competition_Date"]]
                    if later.empty:
                        continue
                    tr = later.iloc[0]
                    # synthesize merge-like row
                    rec = {
                        "College": key[0],
                        "Athlete": key[1],
                        "Season_Year": key[2],
                        "Result_Value_from": fr["Result_Value"],
                        "Result_Value_to": tr["Result_Value"],
                        "World_Athletics_Score_Men_from": fr["World_Athletics_Score_Men"],
                        "World_Athletics_Score_Men_to": tr["World_Athletics_Score_Men"],
                        "Competition_Date_from": fr["Competition_Date"],
                        "Competition_Date_to": tr["Competition_Date"],
                        "Bal_Spec_from": fr.get("Bal_Spec"),
                        "Best_Event_from": fr.get("Best_Event"),
                        "Event_from": from_test,
                        "Event_to": to_test,
                        "WA_Spread_from": fr.get("WA_Spread"),
                    }
                    candidates.append(pd.Series(rec))
                    paired = True
                    break
                if not paired:
                    continue

        for r in candidates:
            if mode == "season_pb":
                x = float(r["Result_Value_from"])
                y = float(r["Result_Value_to"])
                wa_from = float(r["World_Athletics_Score_Men_from"])
                wa_to = float(r["World_Athletics_Score_Men_to"])
                bal = r.get("Bal_Spec_from") or r.get("Bal_Spec_to")
                best = r.get("Best_Event_from") or r.get("Best_Event_to")
                # Prefer group-level fields from either side
                if pd.isna(bal):
                    bal = None
                if pd.isna(best):
                    best = None
                # Map best_event to model naming for labels
                best_model = TEST_TO_MODEL_EVENT.get(best, best) if best else None
                date_from = r.get("Competition_Date_from")
                date_to = r.get("Competition_Date_to")
            else:
                x = float(r["Result_Value_from"])
                y = float(r["Result_Value_to"])
                wa_from = float(r["World_Athletics_Score_Men_from"])
                wa_to = float(r["World_Athletics_Score_Men_to"])
                bal = r.get("Bal_Spec_from")
                best = r.get("Best_Event_from")
                if pd.isna(bal):
                    bal = None
                if pd.isna(best):
                    best = None
                best_model = TEST_TO_MODEL_EVENT.get(best, best) if best else None
                date_from = r.get("Competition_Date_from")
                date_to = r.get("Competition_Date_to")

            # Recompute bal_spec / best_event from PBs in group if missing
            if bal is None or best is None:
                key = (r["College"], r["Athlete"], int(r["Season_Year"]))
                athlete_pb = pb[
                    (pb["College"] == key[0])
                    & (pb["Athlete"] == key[1])
                    & (pb["Season_Year"] == key[2])
                ]
                group_events = (
                    {"100m", "200m", "400m"}
                    if pm.event_group == "Sprints"
                    else {"800m", "1500m", "3000m", "3000m SC", "5000m"}
                )
                gpb = athlete_pb[athlete_pb["Event"].isin(group_events)]
                if not gpb.empty:
                    event_best = gpb.groupby("Event")["World_Athletics_Score_Men"].max()
                    if len(event_best) >= 1:
                        best = event_best.idxmax()
                        best_model = TEST_TO_MODEL_EVENT.get(best, best)
                        if len(event_best) >= 2:
                            spread = float(event_best.max() - event_best.min())
                            bal = "specialized" if spread >= 50 else "balanced"
                        else:
                            bal = "balanced"

            row_from = pd.Series(
                {"Event": from_test, "World_Athletics_Score_Men": wa_from}
            )
            row_to = pd.Series({"Event": to_test, "World_Athletics_Score_Men": wa_to})

            # Pooled prediction
            yhat_pooled = pm.pooled.predict(x)
            # Feature-routed prediction
            label = None
            if pm.recommended_strategy and pm.recommended_strategy != "pooled":
                label = cohort_label(
                    pm.recommended_strategy, row_from, row_to, best_model or "", bal or ""
                )
                # best_event labels in files use best_100m etc with model names
                if label and label.startswith("best_"):
                    pass
                if pm.recommended_strategy == "bal_x_best_event" and bal and best_model:
                    label = f"{bal} ∩ best_{best_model}"
                if pm.recommended_strategy == "best_event" and best_model:
                    label = f"best_{best_model}"

            feat_formula, route = pick_feature_formula(pm, label)
            yhat_feat = feat_formula.predict(x)

            for kind, yhat, route_name in (
                ("pooled", yhat_pooled, f"pooled:{pm.pooled_model_name}"),
                ("feature", yhat_feat, route),
            ):
                if yhat is None:
                    continue
                err = yhat - y
                abs_err = abs(err)
                tol = TOLERANCE.get(to_test)
                rows_out.append(
                    {
                        "mode": mode,
                        "band": pm.band,
                        "event_group": pm.event_group,
                        "from_event": from_test,
                        "to_event": to_test,
                        "college": r["College"],
                        "athlete": r["Athlete"],
                        "season_year": int(r["Season_Year"]),
                        "from_time": x,
                        "to_time_actual": y,
                        "to_time_pred": yhat,
                        "error_sec": err,
                        "abs_error_sec": abs_err,
                        "pct_error": 100.0 * err / y if y else None,
                        "abs_pct_error": 100.0 * abs_err / y if y else None,
                        "within_tolerance": bool(tol is not None and abs_err <= tol),
                        "tolerance_sec": tol,
                        "route_kind": kind,
                        "route_detail": route_name,
                        "cohort_label": label,
                        "bal_spec": bal,
                        "best_event": best,
                        "wa_from": wa_from,
                        "wa_to": wa_to,
                        "model_pooled_name": pm.pooled_model_name,
                        "model_recommended_strategy": pm.recommended_strategy,
                        "train_n": pm.n_train,
                        "train_cv_medae": pm.pooled_cv if kind == "pooled" else pm.recommended_cv,
                        "train_r": pm.r,
                        "date_from": str(date_from)[:10] if date_from is not None else None,
                        "date_to": str(date_to)[:10] if date_to is not None else None,
                        "formula_raw": (pm.pooled.raw if kind == "pooled" else feat_formula.raw),
                    }
                )
    return pd.DataFrame(rows_out)


def summarize(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def agg(g: pd.DataFrame) -> dict:
        e = g["error_sec"].tolist()
        st = err_stats(e)
        st["within_tol_rate"] = float(g["within_tolerance"].mean()) if len(g) else None
        st["median_abs_pct_error"] = float(g["abs_pct_error"].median())
        st["mean_abs_pct_error"] = float(g["abs_pct_error"].mean())
        # correlation predicted vs actual
        if len(g) >= 3:
            st["corr_pred_actual"] = float(np.corrcoef(g["to_time_pred"], g["to_time_actual"])[0, 1])
        else:
            st["corr_pred_actual"] = None
        # vs training CV
        tcv = g["train_cv_medae"].dropna()
        st["train_cv_medae_ref"] = float(tcv.iloc[0]) if len(tcv) else None
        if st.get("medae") is not None and st.get("train_cv_medae_ref"):
            st["medae_minus_train_cv"] = st["medae"] - st["train_cv_medae_ref"]
        else:
            st["medae_minus_train_cv"] = None
        return st

    overall = []
    for (mode, route_kind), g in pred.groupby(["mode", "route_kind"]):
        st = agg(g)
        st.update({"mode": mode, "route_kind": route_kind})
        overall.append(st)
    overall_df = pd.DataFrame(overall)

    by_band = []
    for (mode, band, route_kind), g in pred.groupby(["mode", "band", "route_kind"]):
        st = agg(g)
        st.update({"mode": mode, "band": band, "route_kind": route_kind})
        by_band.append(st)
    by_band_df = pd.DataFrame(by_band)

    by_pair = []
    for keys, g in pred.groupby(
        ["mode", "band", "from_event", "to_event", "route_kind"]
    ):
        st = agg(g)
        st.update(
            {
                "mode": keys[0],
                "band": keys[1],
                "from_event": keys[2],
                "to_event": keys[3],
                "route_kind": keys[4],
                "train_n": int(g["train_n"].iloc[0]),
                "model_recommended_strategy": g["model_recommended_strategy"].iloc[0],
            }
        )
        by_pair.append(st)
    by_pair_df = pd.DataFrame(by_pair).sort_values(
        ["mode", "band", "n", "medae"], ascending=[True, True, False, True]
    )
    return overall_df, by_band_df, by_pair_df


def write_report(
    pred: pd.DataFrame,
    overall: pd.DataFrame,
    by_band: pd.DataFrame,
    by_pair: pd.DataFrame,
    n_models: int,
) -> Path:
    lines = [
        "Outdoor Time-Model Validation — Men's Test Dataset",
        "==================================================",
        "",
        "Models: models/steeplechase (bands 750-950, 800-1000, 850-1050)",
        f"Test data: output/men_test_dataset_results.csv",
        f"Parsed Men pair models: {n_models}",
        "",
        "Evaluation modes",
        "----------------",
        "  1. season_pb: predict target season-best from source season-best",
        "     (matches how the models were trained). Includes supplementary undated",
        "     season-best rows (USI, North Central, Oshkosh, Keiser).",
        "  2. chronological: earliest source mark that precedes a later target mark",
        "     (early-season → later-season check). Dated primary sources only.",
        "",
        "Routes",
        "------",
        "  • pooled: band pooled formula (knn uses reported median ratio fallback)",
        "  • feature: recommended feature cohort when label available; else pooled",
        "",
        "Tolerance: starter absolute seconds from preprocessing design",
        "  (100:0.15, 200:0.30, 400:0.80, 800:1.50, 1500:3.0, 3000:6.0, SC:8.0, 5000:10.0)",
        "",
    ]

    if pred.empty:
        lines.append("No applicable prediction pairs found in the test set.")
        path = OUT_DIR / "validation_report.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    lines.append(f"Total prediction rows: {len(pred)}")
    lines.append(
        f"Unique athlete-seasons tested: "
        f"{pred.groupby(['college','athlete','season_year']).ngroups}"
    )
    lines.append("")

    lines.append("Overall summary")
    lines.append("---------------")
    if not overall.empty:
        show = overall.copy()
        for col in ("mae", "medae", "rmse", "bias_mean", "p90_abs", "within_tol_rate",
                    "median_abs_pct_error", "corr_pred_actual", "medae_minus_train_cv"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: None if pd.isna(v) else round(float(v), 4))
        lines.append(show.to_string(index=False))
    lines.append("")

    lines.append("By band")
    lines.append("-------")
    if not by_band.empty:
        show = by_band.copy()
        for col in ("mae", "medae", "rmse", "within_tol_rate", "median_abs_pct_error",
                    "medae_minus_train_cv"):
            if col in show.columns:
                show[col] = show[col].map(lambda v: None if pd.isna(v) else round(float(v), 4))
        cols = [c for c in ["mode", "band", "route_kind", "n", "medae", "mae", "rmse",
                            "within_tol_rate", "median_abs_pct_error", "medae_minus_train_cv"]
                if c in show.columns]
        lines.append(show[cols].to_string(index=False))
    lines.append("")

    lines.append("By event pair (season_pb, feature route preferred view)")
    lines.append("-------------------------------------------------------")
    sp = by_pair[(by_pair["mode"] == "season_pb") & (by_pair["route_kind"] == "feature")].copy()
    if sp.empty:
        sp = by_pair[(by_pair["mode"] == "season_pb") & (by_pair["route_kind"] == "pooled")].copy()
    if not sp.empty:
        for col in ("medae", "mae", "within_tol_rate", "median_abs_pct_error", "corr_pred_actual",
                    "medae_minus_train_cv", "bias_mean"):
            if col in sp.columns:
                sp[col] = sp[col].map(lambda v: None if pd.isna(v) else round(float(v), 4))
        cols = [c for c in ["band", "from_event", "to_event", "n", "medae", "mae", "bias_mean",
                            "within_tol_rate", "median_abs_pct_error", "corr_pred_actual",
                            "train_cv_medae_ref", "medae_minus_train_cv",
                            "model_recommended_strategy"] if c in sp.columns]
        lines.append(sp[cols].to_string(index=False))
    lines.append("")

    # Best / worst pairs
    lines.append("Closest alignments (season_pb, lowest MedAE, n≥2)")
    lines.append("-------------------------------------------------")
    cand = by_pair[(by_pair["mode"] == "season_pb") & (by_pair["n"] >= 2)].copy()
    if not cand.empty:
        best = cand.sort_values("medae").head(8)
        for _, r in best.iterrows():
            lines.append(
                f"  {r['band']} {r['from_event']}→{r['to_event']} [{r['route_kind']}] "
                f"n={int(r['n'])} MedAE={r['medae']:.3f}s  "
                f"within_tol={r['within_tol_rate']:.0%}  "
                f"vs_train_CV {r.get('medae_minus_train_cv')}"
            )
    lines.append("")
    lines.append("Largest errors (season_pb, highest MedAE, n≥2)")
    lines.append("---------------------------------------------")
    if not cand.empty:
        worst = cand.sort_values("medae", ascending=False).head(8)
        for _, r in worst.iterrows():
            lines.append(
                f"  {r['band']} {r['from_event']}→{r['to_event']} [{r['route_kind']}] "
                f"n={int(r['n'])} MedAE={r['medae']:.3f}s  "
                f"within_tol={r['within_tol_rate']:.0%}"
            )
    lines.append("")

    # Example hits
    lines.append("Example predictions within tolerance (season_pb, feature)")
    lines.append("---------------------------------------------------------")
    hits = pred[
        (pred["mode"] == "season_pb")
        & (pred["route_kind"] == "feature")
        & (pred["within_tolerance"])
    ].sort_values("abs_error_sec").head(12)
    if hits.empty:
        hits = pred[
            (pred["mode"] == "season_pb") & (pred["within_tolerance"])
        ].sort_values("abs_error_sec").head(12)
    for _, r in hits.iterrows():
        lines.append(
            f"  {r['athlete']} ({r['college']}, {r['band']}): "
            f"{r['from_event']} {r['from_time']:.2f}s → pred {r['to_event']} "
            f"{r['to_time_pred']:.2f}s vs actual {r['to_time_actual']:.2f}s "
            f"(|err|={r['abs_error_sec']:.3f}s)"
        )
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append("  • knn_median pooled models fall back to the report's median ratio.")
    lines.append("  • Feature route uses cohort formula when the athlete's label exists;")
    lines.append("    otherwise falls back to pooled.")
    lines.append("  • Test athletes are NCAA D1; models were fit on club data — expect")
    lines.append("    some distribution shift.")
    lines.append("  • Positive bias => model predicts slower than actual.")
    lines.append("")

    path = OUT_DIR / "validation_report.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_validate() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TEST_CSV)
    pb = load_test_pb(df)

    all_models: list[PairModel] = []
    for band, fname in BAND_FILES.items():
        path = MODELS_DIR / fname
        parsed = parse_band_file(path, band)
        print(f"Parsed {band}: {len(parsed)} Men pair models from {fname}")
        all_models.extend(parsed)

    # Quick parse health check
    kinds = {}
    for m in all_models:
        kinds[m.pooled.kind] = kinds.get(m.pooled.kind, 0) + 1
    print("Pooled formula kinds:", kinds)

    pred_pb = evaluate(all_models, df, pb, "season_pb")
    pred_chr = evaluate(all_models, df, pb, "chronological")
    pred = pd.concat([pred_pb, pred_chr], ignore_index=True)

    overall, by_band, by_pair = summarize(pred)

    pred_path = OUT_DIR / "predictions.csv"
    overall_path = OUT_DIR / "summary_overall.csv"
    band_path = OUT_DIR / "summary_by_band.csv"
    pair_path = OUT_DIR / "summary_by_pair.csv"
    pred.to_csv(pred_path, index=False)
    overall.to_csv(overall_path, index=False)
    by_band.to_csv(band_path, index=False)
    by_pair.to_csv(pair_path, index=False)

    report = write_report(pred, overall, by_band, by_pair, len(all_models))
    print(f"Wrote {pred_path} ({len(pred)} rows)")
    print(f"Wrote {overall_path}")
    print(f"Wrote {band_path}")
    print(f"Wrote {pair_path}")
    print(f"Wrote {report}")
    print()
    print(report.read_text())

    # Inferential tests + research figures
    from research_stats import run_research_stats

    print("\n--- Research statistics & figures ---\n")
    run_research_stats(pred)


def main() -> None:
    run_validate()


if __name__ == "__main__":
    main()

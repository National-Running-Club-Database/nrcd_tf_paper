"""Indoor → outdoor event-pair time/mark prediction models.

For each gender and requested (indoor from_event → outdoor to_event) pair,
fit a linear model on athletes with season bests in both, then predict the
outdoor mark from the indoor mark for all athletes with an indoor from-event.

Season window: 2024–2026. Same athlete × calendar year (indoor season year
aligned to outdoor season year).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

FIELD_EVENT_IDS = {38, 39, 40, 41, 42, 43, 44, 45, 46}


def parse_performance(result_time: str, event_id: int) -> float:
    raw = (result_time or "").strip().lower().replace("m", "")
    if not raw:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")
    try:
        if ":" in raw:
            mins, secs = raw.split(":", 1)
            return float(mins) * 60 + float(secs)
        return float(raw)
    except ValueError:
        return float("inf") if event_id not in FIELD_EVENT_IDS else float("-inf")


RELAY_IDS = {21, 22, 24, 26, 29, 30, 31}
SEASONS = (2024, 2025, 2026)
MIN_N = 10

# Indoor event_id → label
INDOOR_EVENTS = {
    2: "Indoor 60m",
    4: "Indoor 200m",
    6: "Indoor 400m",
    9: "Indoor 800m",
    13: "Indoor Mile",
    14: "Indoor 3000m",
    17: "Indoor 5000m",
    33: "Indoor 60m Hurdles",
    38: "Indoor Long Jump",
    39: "Indoor Triple Jump",
    40: "Indoor High Jump",
    41: "Indoor Shot Put",
}

OUTDOOR_EVENTS = {
    3: "Outdoor 100m",
    4: "Outdoor 200m",
    6: "Outdoor 400m",
    9: "Outdoor 800m",
    11: "Outdoor 1500m",
    17: "Outdoor 5000m",
    20: "Outdoor 3000m Steeplechase",
    34: "Outdoor 100m Hurdles",
    35: "Outdoor 110m Hurdles",
    38: "Outdoor Long Jump",
    39: "Outdoor Triple Jump",
    40: "Outdoor High Jump",
    41: "Outdoor Shot Put",
}

# (group, indoor_eid, outdoor_eid) — gender-specific hurdles handled in PAIRS
PAIRS: list[tuple[str, int, int, str | None]] = [
    # group, indoor_id, outdoor_id, gender_filter (None = both)
    ("Sprints", 2, 3, None),
    ("Sprints", 4, 4, None),
    ("Sprints", 6, 6, None),
    ("Distance", 9, 9, None),
    ("Distance", 13, 11, None),
    ("Distance", 14, 17, None),
    ("Distance", 14, 11, None),
    ("Distance", 14, 20, None),
    ("Distance", 17, 17, None),
    ("Jumps", 38, 38, None),
    ("Jumps", 39, 39, None),
    ("Jumps", 40, 40, None),
    ("Hurdles", 33, 34, None),  # 60H → 100H (primarily women; fit both if n≥10)
    ("Hurdles", 33, 35, None),  # 60H → 110H (primarily men; fit both if n≥10)
    ("Throws", 41, 41, None),
]

DISCIPLINE_INDOOR_FOLDER = {
    "Sprints": "Indoor_Sprints",
    "Distance": "Indoor_Distance",
    "Hurdles": "Indoor_Hurdles",
    "Jumps": "Indoor_Jumps",
    "Throws": "Indoor_Throws",
}


def outdoor_paths(group: str, gender: str, year: int) -> list[Path]:
    g = gender
    if group == "Sprints":
        return [
            REPO
            / "non_relays_findings"
            / "Sprints_Events_Counting"
            / f"Sprinters_{g}_Outdoor_{year}_Data.csv"
        ]
    if group == "Distance":
        return [
            REPO
            / "new_steeplechase_data"
            / "Distance_Relays_Findings"
            / f"Relays_Distance_{g}_Outdoor_{year}_Data.csv",
            REPO
            / "non_relays_findings"
            / "Distance_Events_Counting"
            / f"Distance_{g}_Outdoor_{year}_Data.csv",
        ]
    if group == "Hurdles":
        return [
            REPO
            / "non_relays_findings"
            / "Hurdles_Events_Counting"
            / f"Hurdles_{g}_Outdoor_{year}_Data.csv"
        ]
    if group == "Jumps":
        return [
            REPO
            / "non_relays_findings"
            / "Jumps_Events_Counting"
            / f"Jumps_{g}_Outdoor_{year}_Data.csv"
        ]
    if group == "Throws":
        return [
            REPO
            / "non_relays_findings"
            / "Throws_Events_Counting"
            / f"Throws_{g}_Outdoor_{year}_Data.csv"
        ]
    return []


def indoor_path(group: str, gender: str, year: int) -> Path:
    folder = DISCIPLINE_INDOOR_FOLDER[group]
    return (
        REPO
        / "indoor_analysis"
        / folder
        / f"Indoor_Relays_{group}_{gender}_{year}_Data.csv"
    )


def is_field(event_id: int) -> bool:
    return event_id in FIELD_EVENT_IDS


def better(a: float, b: float, field: bool) -> bool:
    """True if a is a better performance than b."""
    if field:
        return a > b
    return a < b


def format_mark(value: float, field: bool) -> str:
    if field:
        return f"{value:.2f}m"
    if value >= 60:
        mins = int(value // 60)
        secs = value - mins * 60
        return f"{mins}:{secs:05.2f}"
    return f"{value:.2f}"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def normalize(df: pd.DataFrame, event_ids: set[int], season: int) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    if "athlete_id_2" in d.columns:
        d = d[d["athlete_id_2"].isna()].copy()
    d["running_event_id"] = pd.to_numeric(d["running_event_id"], errors="coerce")
    d = d.dropna(subset=["running_event_id"])
    d["running_event_id"] = d["running_event_id"].astype(int)
    d = d[~d["running_event_id"].isin(RELAY_IDS)]
    d = d[d["running_event_id"].isin(event_ids)]
    d["athlete_id"] = pd.to_numeric(d["athlete_id"], errors="coerce")
    d = d.dropna(subset=["athlete_id"])
    d["athlete_id"] = d["athlete_id"].astype(int)
    d["season_year"] = season
    return d


def season_bests(
    df: pd.DataFrame, event_id: int
) -> dict[tuple[int, int], float]:
    """(athlete_id, season_year) -> best mark."""
    field = is_field(event_id)
    bests: dict[tuple[int, int], float] = {}
    sub = df[df["running_event_id"] == event_id]
    for r in sub.itertuples():
        mark = parse_performance(str(r.result_time), event_id)
        if field:
            if math.isinf(mark) or mark <= 0:
                continue
        else:
            if math.isinf(mark) or mark <= 0:
                continue
        key = (int(r.athlete_id), int(r.season_year))
        cur = bests.get(key)
        if cur is None or better(mark, cur, field):
            bests[key] = mark
    return bests


def load_group_frames(group: str, gender: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    indoor_ids = {p[1] for p in PAIRS if p[0] == group and (p[3] is None or p[3] == gender)}
    outdoor_ids = {p[2] for p in PAIRS if p[0] == group and (p[3] is None or p[3] == gender)}
    in_parts, out_parts = [], []
    for year in SEASONS:
        inn = normalize(load_csv(indoor_path(group, gender, year)), indoor_ids, year)
        if not inn.empty:
            in_parts.append(inn)
        for path in outdoor_paths(group, gender, year):
            out = normalize(load_csv(path), outdoor_ids, year)
            if not out.empty:
                out_parts.append(out)
                break
    indoor = pd.concat(in_parts, ignore_index=True) if in_parts else pd.DataFrame()
    outdoor = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    return indoor, outdoor


def linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float, float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / n)
    den_r = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    r = num / den_r if den_r else 0.0
    mae = statistics.median(abs(r) for r in resid)
    return intercept, slope, r, rmse, mae


@dataclass
class PairModel:
    gender: str
    event_group: str
    from_event: str
    to_event: str
    from_id: int
    to_id: int
    n: int
    intercept: float
    slope: float
    r: float
    rmse: float
    medae: float
    ratio_median: float
    unit: str  # seconds | meters

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x

    def formula(self) -> str:
        u = "s" if self.unit == "seconds" else "m"
        if self.intercept >= 0:
            return (
                f"{self.to_event} = {self.intercept:.4f} + {self.slope:.4f} × "
                f"{self.from_event}  ({u})"
            )
        return (
            f"{self.to_event} = {self.slope:.4f} × {self.from_event} − "
            f"{abs(self.intercept):.4f}  ({u})"
        )


def fit_and_predict(
    gender: str,
    group: str,
    indoor_id: int,
    outdoor_id: int,
    indoor_df: pd.DataFrame,
    outdoor_df: pd.DataFrame,
) -> tuple[PairModel | None, pd.DataFrame]:
    from_name = INDOOR_EVENTS[indoor_id]
    to_name = OUTDOOR_EVENTS[outdoor_id]
    field = is_field(outdoor_id)
    unit = "meters" if field else "seconds"

    in_best = season_bests(indoor_df, indoor_id)
    out_best = season_bests(outdoor_df, outdoor_id)

    paired_keys = sorted(set(in_best) & set(out_best))
    xs = [in_best[k] for k in paired_keys]
    ys = [out_best[k] for k in paired_keys]

    if len(paired_keys) < MIN_N:
        # Still emit empty predictions note
        model = None
        pred_rows = []
        for key, x in in_best.items():
            aid, year = key
            pred_rows.append(
                {
                    "gender": gender,
                    "event_group": group,
                    "athlete_id": aid,
                    "season_year": year,
                    "from_event": from_name,
                    "to_event": to_name,
                    "indoor_mark": round(x, 4),
                    "indoor_mark_fmt": format_mark(x, field),
                    "predicted_outdoor": None,
                    "predicted_outdoor_fmt": None,
                    "actual_outdoor": round(out_best[key], 4) if key in out_best else None,
                    "actual_outdoor_fmt": (
                        format_mark(out_best[key], field) if key in out_best else None
                    ),
                    "abs_error": None,
                    "pct_error": None,
                    "in_training_pair": key in out_best,
                    "model_fit": False,
                    "n_model": len(paired_keys),
                }
            )
        return model, pd.DataFrame(pred_rows)

    intercept, slope, r, rmse, medae = linreg(xs, ys)
    ratios = [y / x for x, y in zip(xs, ys) if x > 0]
    ratio_med = statistics.median(ratios) if ratios else float("nan")

    model = PairModel(
        gender=gender,
        event_group=group,
        from_event=from_name,
        to_event=to_name,
        from_id=indoor_id,
        to_id=outdoor_id,
        n=len(paired_keys),
        intercept=intercept,
        slope=slope,
        r=r,
        rmse=rmse,
        medae=medae,
        ratio_median=ratio_med,
        unit=unit,
    )

    pred_rows = []
    for key, x in in_best.items():
        aid, year = key
        yhat = model.predict(x)
        y_act = out_best.get(key)
        abs_err = abs(yhat - y_act) if y_act is not None else None
        pct_err = (
            100.0 * abs(yhat - y_act) / y_act
            if y_act is not None and y_act != 0
            else None
        )
        pred_rows.append(
            {
                "gender": gender,
                "event_group": group,
                "athlete_id": aid,
                "season_year": year,
                "from_event": from_name,
                "to_event": to_name,
                "indoor_mark": round(x, 4),
                "indoor_mark_fmt": format_mark(x, field),
                "predicted_outdoor": round(yhat, 4),
                "predicted_outdoor_fmt": format_mark(yhat, field),
                "actual_outdoor": round(y_act, 4) if y_act is not None else None,
                "actual_outdoor_fmt": format_mark(y_act, field) if y_act is not None else None,
                "abs_error": round(abs_err, 4) if abs_err is not None else None,
                "pct_error": round(pct_err, 4) if pct_err is not None else None,
                "in_training_pair": y_act is not None,
                "model_fit": True,
                "n_model": model.n,
                "formula": model.formula(),
            }
        )
    return model, pd.DataFrame(pred_rows)


def run() -> None:
    all_models: list[PairModel] = []
    all_preds: list[pd.DataFrame] = []

    for gender in ("Men", "Women"):
        for group in ("Sprints", "Distance", "Jumps", "Hurdles", "Throws"):
            pairs = [
                p for p in PAIRS if p[0] == group and (p[3] is None or p[3] == gender)
            ]
            if not pairs:
                continue
            print(f"[Indoor→Outdoor] Loading {gender} {group}…")
            indoor_df, outdoor_df = load_group_frames(group, gender)
            print(
                f"  indoor rows={len(indoor_df):,} outdoor rows={len(outdoor_df):,}"
            )
            for _, iid, oid, _ in pairs:
                model, preds = fit_and_predict(
                    gender, group, iid, oid, indoor_df, outdoor_df
                )
                if model is None:
                    print(
                        f"  SKIP {INDOOR_EVENTS[iid]} → {OUTDOOR_EVENTS[oid]} "
                        f"(n<{MIN_N})"
                    )
                else:
                    print(
                        f"  OK   {model.from_event} → {model.to_event} "
                        f"n={model.n} r={model.r:.3f} MedAE={model.medae:.3f}"
                    )
                    all_models.append(model)
                if not preds.empty:
                    all_preds.append(preds)

    models_df = pd.DataFrame(
        [
            {
                "gender": m.gender,
                "event_group": m.event_group,
                "from_event": m.from_event,
                "to_event": m.to_event,
                "from_event_id": m.from_id,
                "to_event_id": m.to_id,
                "n_athletes": m.n,
                "intercept": round(m.intercept, 6),
                "slope": round(m.slope, 6),
                "r": round(m.r, 4),
                "rmse": round(m.rmse, 4),
                "median_abs_error": round(m.medae, 4),
                "ratio_median": round(m.ratio_median, 4),
                "unit": m.unit,
                "formula": m.formula(),
            }
            for m in all_models
        ]
    )
    models_path = OUT / "indoor_to_outdoor_models.csv"
    models_df.to_csv(models_path, index=False)

    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    preds_path = OUT / "indoor_to_outdoor_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Per-pair summary with held-in sample errors
    summary_rows = []
    for m in all_models:
        sub = preds_df[
            (preds_df["gender"] == m.gender)
            & (preds_df["from_event"] == m.from_event)
            & (preds_df["to_event"] == m.to_event)
            & (preds_df["in_training_pair"] == True)  # noqa: E712
            & preds_df["abs_error"].notna()
        ]
        summary_rows.append(
            {
                "gender": m.gender,
                "event_group": m.event_group,
                "pair": f"{m.from_event} → {m.to_event}",
                "n": m.n,
                "r": round(m.r, 4),
                "rmse": round(m.rmse, 4),
                "medae": round(m.medae, 4),
                "mean_abs_pct_error": (
                    round(float(sub["pct_error"].mean()), 3) if len(sub) else None
                ),
                "formula": m.formula(),
                "n_predicted_with_indoor_only": int(
                    (
                        (preds_df["gender"] == m.gender)
                        & (preds_df["from_event"] == m.from_event)
                        & (preds_df["to_event"] == m.to_event)
                        & (preds_df["in_training_pair"] == False)  # noqa: E712
                    ).sum()
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "indoor_to_outdoor_summary.csv", index=False)

    write_findings(models_df, summary_df, preds_df)
    print(f"[Indoor→Outdoor] Wrote {OUT}")


def write_findings(
    models_df: pd.DataFrame, summary_df: pd.DataFrame, preds_df: pd.DataFrame
) -> None:
    lines = [
        "Indoor → Outdoor Event Predictions",
        "=" * 72,
        "",
        "Question",
        "--------",
        "For each gender and event pair below, predict outdoor (to_event) performance",
        "from indoor (from_event) season best using a linear model fit on athletes",
        "with both marks in the same year (2024–2026).",
        "",
        "Model: outdoor = intercept + slope × indoor",
        "Track events: times in seconds (lower better). Field: marks in meters (higher better).",
        "Indoor 60m = running_event_id 2; Indoor 60m Hurdles = 33.",
        "",
        "Pairs requested",
        "---------------",
        "Sprints: 60m→100m, 200m→200m, 400m→400m",
        "Distance: 800→800, Mile→1500, 3000→5000, 3000→1500, 3000→Steeple, 5000→5000",
        "Jumps: LJ→LJ, TJ→TJ, HJ→HJ",
        "Hurdles: Women 60H→100H; Men 60H→110H",
        "Throws: SP→SP",
        "",
    ]

    for gender in ("Men", "Women"):
        lines.append(f"{gender}")
        lines.append("-" * 40)
        sub = summary_df[summary_df["gender"] == gender]
        if sub.empty:
            lines.append("  (no models fit)")
            lines.append("")
            continue
        for _, r in sub.iterrows():
            mape = r["mean_abs_pct_error"]
            mape_s = f"{mape:.2f}%" if pd.notna(mape) else "n/a"
            lines.append(f"  {r['pair']}")
            lines.append(
                f"    n={int(r['n'])}  r={r['r']:.3f}  MedAE={r['medae']:.3f}  "
                f"MAPE={mape_s}  indoor-only preds={int(r['n_predicted_with_indoor_only'])}"
            )
            lines.append(f"    {r['formula']}")
            lines.append("")

    # Example predictions: median indoor athlete per pair
    lines += [
        "Example predictions (median indoor mark in paired sample)",
        "-" * 40,
    ]
    for _, m in models_df.iterrows():
        sub = preds_df[
            (preds_df["gender"] == m["gender"])
            & (preds_df["from_event"] == m["from_event"])
            & (preds_df["to_event"] == m["to_event"])
            & (preds_df["in_training_pair"] == True)  # noqa: E712
        ]
        if sub.empty:
            continue
        mid = sub.sort_values("indoor_mark").iloc[len(sub) // 2]
        lines.append(
            f"  {m['gender']} {m['from_event']} {mid['indoor_mark_fmt']} → "
            f"pred {mid['predicted_outdoor_fmt']}"
            + (
                f" (actual {mid['actual_outdoor_fmt']})"
                if pd.notna(mid.get("actual_outdoor"))
                else ""
            )
        )

    lines += [
        "",
        "Artifacts",
        "---------",
        "  output/indoor_to_outdoor_models.csv      — formulas by gender × pair",
        "  output/indoor_to_outdoor_predictions.csv — athlete-level predictions",
        "  output/indoor_to_outdoor_summary.csv     — fit quality summary",
        "  output/indoor_to_outdoor_findings.txt",
        "  output/Summary_findings.md",
        "",
        "Run: python time_models/Indoor_to_Outdoor_Predictions/main.py",
        "",
    ]
    (OUT / "indoor_to_outdoor_findings.txt").write_text("\n".join(lines) + "\n")

    md = [
        "# Indoor → Outdoor Predictions — Summary Findings",
        "",
        "Linear models predict outdoor season-best from indoor season-best for the "
        "requested event pairs (2024–2026, same athlete-year).",
        "",
        "## Models by gender",
        "",
    ]
    for gender in ("Men", "Women"):
        md.append(f"### {gender}")
        md.append("")
        md.append("| Pair | n | r | MedAE | MAPE | Formula |")
        md.append("|------|---|---|-------|------|---------|")
        sub = summary_df[summary_df["gender"] == gender]
        for _, r in sub.iterrows():
            mape = r["mean_abs_pct_error"]
            mape_s = f"{mape:.2f}%" if pd.notna(mape) else "—"
            md.append(
                f"| {r['pair']} | {int(r['n'])} | {r['r']:.3f} | {r['medae']:.3f} | "
                f"{mape_s} | `{r['formula']}` |"
            )
        md.append("")

    md += [
        "## Artifacts",
        "",
        "- `output/indoor_to_outdoor_models.csv`",
        "- `output/indoor_to_outdoor_predictions.csv`",
        "- `output/indoor_to_outdoor_summary.csv`",
        "- `output/indoor_to_outdoor_findings.txt`",
        "",
    ]
    (OUT / "Summary_findings.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT / 'indoor_to_outdoor_findings.txt'}")
    print(f"Wrote {OUT / 'Summary_findings.md'}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()

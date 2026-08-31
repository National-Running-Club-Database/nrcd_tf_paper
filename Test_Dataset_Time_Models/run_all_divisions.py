"""All_Divisions preprocess + validate + research pipeline.

Uses every men's document scraper (NCAA D1/D3 + NAIA): original test-set
schools plus newly added D1 documents. Season-PB protocol only; same absolute
tolerances as the main validation. All artifacts live under output/All_Divisions/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from d1_new_scrapers import NEW_D1_PRIMARY, NEW_D1_SUPPLEMENTARY
from preprocess import (
    OUT_DIR as ROOT_OUT,
    add_features,
    athlete_feature_summary,
    load_coefficients,
    scrape_depaul,
    scrape_keiser_supplementary,
    scrape_north_central_supplementary,
    scrape_north_florida,
    scrape_oshkosh_supplementary,
    scrape_siue,
    scrape_usi_supplementary,
)

FOLDER = "All_Divisions"
OUT_DIR = ROOT_OUT / FOLDER


def run_preprocess_all_divisions() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeffs = load_coefficients()

    primary_scrapers = [
        ("SIUE", scrape_siue),
        ("North Florida", scrape_north_florida),
        ("DePaul", scrape_depaul),
        *NEW_D1_PRIMARY,
    ]
    supplementary_scrapers = [
        ("Southern Indiana (suppl.)", scrape_usi_supplementary),
        ("North Central (suppl.)", scrape_north_central_supplementary),
        ("Wisconsin-Oshkosh (suppl.)", scrape_oshkosh_supplementary),
        ("Keiser (suppl.)", scrape_keiser_supplementary),
        *NEW_D1_SUPPLEMENTARY,
    ]

    all_rows: list[dict] = []
    counts: dict[str, int] = {}
    for label, fn in primary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"[All_Divisions] Scraped {label}: {len(part)} dated results")

    for label, fn in supplementary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"[All_Divisions] Scraped {label}: {len(part)} season-best rows")

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise SystemExit("All_Divisions: no rows scraped")

    before = len(raw)
    raw = raw.drop_duplicates(
        subset=["College", "Athlete", "Event", "Result", "Competition_Date", "Meet"]
    )
    print(f"[All_Divisions] Deduped {before - len(raw)} rows; {len(raw)} remain")

    # Drop known Air Force PDF mislabel if any residual Colorado State rows appear
    raw = raw[raw["College"].astype(str).str.lower() != "colorado state"].copy()

    featured = add_features(raw)
    summary = athlete_feature_summary(featured)

    core_cols = [
        "Athlete",
        "College",
        "College_Division",
        "Event",
        "Event_Group",
        "Result",
        "Result_Value",
        "Indoor_Outdoor",
        "Competition_Date",
        "Meet",
        "Place",
        "Wind",
        "World_Athletics_Score_Men",
        "World_Athletics_Score_Women",
        "Season_Year",
        "Result_Order_In_Season",
        "Season_Day_Index",
        "Is_First_Result_Of_Season",
        "Is_Season_Event_PB",
        "WA_Spread",
        "Bal_Spec",
        "Best_Event",
        "Best_Is_This_Event",
        "Events_Competed_In_Group",
        "Events_Competed_Bucket",
        "Tolerance",
        "Tolerance_Unit",
        "Source_File",
        "Source_Role",
        "Gender",
    ]
    featured = featured[[c for c in core_cols if c in featured.columns]]

    results_path = OUT_DIR / "men_test_dataset_results.csv"
    features_path = OUT_DIR / "men_athlete_season_features.csv"
    suppl_path = OUT_DIR / "men_supplementary_season_pb.csv"
    readme_path = OUT_DIR / "README.txt"

    featured.to_csv(results_path, index=False)
    summary.to_csv(features_path, index=False)
    suppl = featured[featured["Source_Role"] == "supplementary_season_pb"]
    suppl.to_csv(suppl_path, index=False)

    n_primary = int((featured["Source_Role"] == "primary_dated").sum())
    n_suppl = int((featured["Source_Role"] == "supplementary_season_pb").sum())
    colleges = sorted(featured["College"].dropna().unique())
    div_counts = (
        featured.groupby("College_Division")["College"]
        .nunique()
        .sort_values(ascending=False)
        .to_dict()
        if "College_Division" in featured.columns
        else {}
    )

    lines = [
        "All_Divisions Men's Test Dataset — Time Models",
        "==============================================",
        "",
        "Filter: none (NCAA D1 / D3 / NAIA and all scraped men's documents).",
        "Tolerances: same starter absolute-second / metre bands as main pipeline.",
        "Protocol for validation: season-PB only.",
        "",
        "Primary (dated) sources:",
    ]
    for label, _ in primary_scrapers:
        lines.append(f"  • {label}: {counts.get(label, 0)} rows")
    lines.append("")
    lines.append("Supplementary (undated season-best) sources:")
    for label, _ in supplementary_scrapers:
        lines.append(f"  • {label}: {counts.get(label, 0)} rows")
    lines += [
        "",
        f"Colleges: {', '.join(colleges)}",
        f"Schools by division: {div_counts}",
        f"Total rows: {len(featured)} (primary_dated={n_primary}, supplementary_season_pb={n_suppl})",
        f"Athletes×season: {featured.groupby(['College', 'Athlete', 'Season_Year']).ngroups}",
        "",
        f"Outputs: {results_path.name}, {features_path.name}, {suppl_path.name}",
    ]
    readme_path.write_text("\n".join(lines) + "\n")
    print(f"[All_Divisions] Wrote {results_path}")
    print(f"[All_Divisions] Wrote {features_path}")
    print(f"[All_Divisions] Wrote {suppl_path}")
    print(f"[All_Divisions] Wrote {readme_path}")
    return results_path


def run_validate_all_divisions(
    *,
    artifact_subdir: str | None = None,
    min_results_per_event: int = 1,
    min_results_from: int | None = None,
    min_results_to: int | None = None,
    findings_filename: str = "Summary_findings.md",
) -> None:
    """Season-PB validation on all-division men's data.

    artifact_subdir: optional folder under All_Divisions/ for outputs
      (e.g. \"2+_Results\"). When None, writes to All_Divisions/model_validation/.
    min_results_per_event: legacy shorthand for both sides when from/to unset.
    min_results_from / min_results_to: require ≥N results in A and/or B.
    findings_filename: markdown findings file name under artifact_root.
    """
    import validate as V

    n_from = min_results_per_event if min_results_from is None else min_results_from
    n_to = min_results_per_event if min_results_to is None else min_results_to

    test_csv = OUT_DIR / "men_test_dataset_results.csv"
    if not test_csv.exists():
        raise SystemExit(f"Missing {test_csv}; run preprocess first")

    artifact_root = OUT_DIR / artifact_subdir if artifact_subdir else OUT_DIR
    out_dir = artifact_root / "model_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"All_Divisions/{artifact_subdir}" if artifact_subdir else "All_Divisions"

    old_test, old_out = V.TEST_CSV, V.OUT_DIR
    V.TEST_CSV = test_csv
    V.OUT_DIR = out_dir

    try:
        df = pd.read_csv(test_csv)
        if "Gender" in df.columns:
            df = df[df["Gender"].astype(str).str.lower().isin(["men", "m", "male"])].copy()
        df = df[df["College"].astype(str).str.lower() != "colorado state"].copy()

        print(
            f"[{tag}] Validation frame: {len(df)} rows | "
            f"min_results A≥{n_from} B≥{n_to} | "
            f"divisions={sorted(df['College_Division'].dropna().unique()) if 'College_Division' in df.columns else 'n/a'} | "
            f"colleges={sorted(df['College'].dropna().unique())}"
        )

        pb = V.load_test_pb(df)

        all_models: list = []
        for band, fname in V.BAND_FILES.items():
            path = V.MODELS_DIR / fname
            parsed = V.parse_band_file(path, band)
            print(f"[{tag}] Parsed {band}: {len(parsed)} Men pair models")
            all_models.extend(parsed)

        pred = V.evaluate(
            all_models,
            df,
            pb,
            "season_pb",
            min_results_from=n_from,
            min_results_to=n_to,
        )
        overall, by_band, by_pair = V.summarize(pred)

        pred_path = out_dir / "predictions.csv"
        overall_path = out_dir / "summary_overall.csv"
        band_path = out_dir / "summary_by_band.csv"
        pair_path = out_dir / "summary_by_pair.csv"
        pred.to_csv(pred_path, index=False)
        overall.to_csv(overall_path, index=False)
        by_band.to_csv(band_path, index=False)
        by_pair.to_csv(pair_path, index=False)

        report_path = out_dir / "validation_report.txt"
        tmp = V.write_report(pred, overall, by_band, by_pair, len(all_models))
        extra = (
            f"{tag} — all scraped men's documents · SEASON-PB protocol only\n"
            "Includes NCAA D1 / D3 / NAIA. Same absolute tolerances as main pipeline.\n"
            "Chronological evaluation intentionally omitted (matches D1_Only analysis).\n"
        )
        if n_from > 1 or n_to > 1:
            extra += (
                f"Filter: predict A→B only if athlete-season has ≥{n_from} results "
                f"in event A and ≥{n_to} results in event B.\n"
            )
        extra += (
            "Note: 2026_Outdoor_Marks.pdf is Air Force (GOAIRFORCEFALCONS.COM), "
            "not Colorado State; men pages 1–4 only.\n\n"
        )
        report_path.write_text(extra + tmp.read_text())

        print(f"[{tag}] Wrote {pred_path} ({len(pred)} rows)")
        print(f"[{tag}] Wrote {overall_path}")
        print(f"[{tag}] Wrote {band_path}")
        print(f"[{tag}] Wrote {pair_path}")
        print(f"[{tag}] Wrote {report_path}")
        print()
        print(report_path.read_text()[:2800])

        print(f"\n--- {tag} research statistics (season_pb) ---\n")
        _run_research(pred, artifact_root=artifact_root, tag=tag)
        _write_success_tables(pred, artifact_root=artifact_root, tag=tag)
        _write_summary_findings(
            pred,
            df,
            artifact_root=artifact_root,
            tag=tag,
            min_results_from=n_from,
            min_results_to=n_to,
            findings_filename=findings_filename,
            artifact_subdir=artifact_subdir,
        )
    finally:
        V.TEST_CSV = old_test
        V.OUT_DIR = old_out


def _run_research(
    pred: pd.DataFrame,
    *,
    artifact_root: Path | None = None,
    tag: str = "All_Divisions",
) -> None:
    import research_stats as R

    root = artifact_root or OUT_DIR
    research_dir = root / "model_validation" / "research"
    plot_dir = research_dir / "plots"
    research_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    old_pred, old_out, old_plot = R.PRED_PATH, R.OUT_DIR, R.PLOT_DIR
    R.PRED_PATH = root / "model_validation" / "predictions.csv"
    R.OUT_DIR = research_dir
    R.PLOT_DIR = plot_dir
    try:
        R.run_research_stats(pred)
    finally:
        R.PRED_PATH = old_pred
        R.OUT_DIR = old_out
        R.PLOT_DIR = old_plot


def _write_success_tables(
    pred: pd.DataFrame,
    *,
    artifact_root: Path | None = None,
    tag: str = "All_Divisions",
) -> None:
    """Band×pair and athlete-level success rates (feature route)."""
    root = artifact_root or OUT_DIR
    out_dir = root / "model_validation"
    f = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    if f.empty:
        print(f"[{tag}] No feature-route predictions; skip success tables")
        return

    band_rows = []
    for (band, a, b), g in f.groupby(["band", "from_event", "to_event"]):
        band_rows.append(
            {
                "band": band,
                "from_event": a,
                "to_event": b,
                "n": len(g),
                "successes": int(g["within_tolerance"].sum()),
                "success_rate": float(g["within_tolerance"].mean()),
                "medape": float(g["abs_pct_error"].median()),
                "medae": float(g["abs_error_sec"].median()),
                "strategy": g["model_recommended_strategy"].iloc[0]
                if "model_recommended_strategy" in g.columns
                else None,
            }
        )
    by_band = pd.DataFrame(band_rows).sort_values(
        ["band", "success_rate", "n"], ascending=[True, False, False]
    )
    path_band = out_dir / "success_rate_by_band_pair.csv"
    by_band.to_csv(path_band, index=False)
    print(f"[{tag}] Wrote {path_band}")

    keys = ["college", "athlete", "season_year"]
    ath_rows = []
    for (a, b), g in f.groupby(["from_event", "to_event"]):
        ath = g.groupby(keys)["within_tolerance"].any()
        ath_rows.append(
            {
                "from_event": a,
                "to_event": b,
                "n_athletes": len(ath),
                "n_successful_athletes": int(ath.sum()),
                "athlete_success_rate": float(ath.mean()),
                "n_preds": len(g),
                "row_success_rate": float(g["within_tolerance"].mean()),
                "medape": float(g["abs_pct_error"].median()),
            }
        )
    by_ath = pd.DataFrame(ath_rows).sort_values(
        ["athlete_success_rate", "n_athletes"], ascending=[False, False]
    )
    path_ath = out_dir / "athlete_success_rate_by_pair.csv"
    by_ath.to_csv(path_ath, index=False)
    print(f"[{tag}] Wrote {path_ath}")


def _write_summary_findings(
    pred: pd.DataFrame,
    df: pd.DataFrame,
    *,
    artifact_root: Path | None = None,
    tag: str = "All_Divisions",
    min_results_from: int = 1,
    min_results_to: int = 1,
    findings_filename: str = "Summary_findings.md",
    artifact_subdir: str | None = None,
) -> None:
    root = artifact_root or OUT_DIR
    f = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "feature")].copy()
    p = pred[(pred["mode"] == "season_pb") & (pred["route_kind"] == "pooled")].copy()
    overall = root / "model_validation" / "summary_overall.csv"
    ov = pd.read_csv(overall) if overall.exists() else pd.DataFrame()

    n_primary = int((df["Source_Role"] == "primary_dated").sum()) if "Source_Role" in df.columns else len(df)
    n_suppl = int((df["Source_Role"] == "supplementary_season_pb").sum()) if "Source_Role" in df.columns else 0
    colleges = sorted(df["College"].dropna().unique())
    divs = (
        sorted(df["College_Division"].dropna().unique())
        if "College_Division" in df.columns
        else []
    )
    n_ath = f.groupby(["college", "athlete", "season_year"]).ngroups if not f.empty else 0

    def feat_row(route: str) -> dict:
        sub = ov[(ov["mode"] == "season_pb") & (ov["route_kind"] == route)]
        if sub.empty:
            return {}
        r = sub.iloc[0]
        return {
            "n": int(r["n"]),
            "medape": float(r["median_abs_pct_error"]),
            "tol": float(r["within_tol_rate"]),
        }

    fr, pr = feat_row("feature"), feat_row("pooled")

    band_csv = root / "model_validation" / "success_rate_by_band_pair.csv"
    ath_csv = root / "model_validation" / "athlete_success_rate_by_pair.csv"
    by_band = pd.read_csv(band_csv) if band_csv.exists() else pd.DataFrame()
    by_ath = pd.read_csv(ath_csv) if ath_csv.exists() else pd.DataFrame()

    filter_line = "**Filter:** none — all scraped men's documents (NCAA D1 / D3 / NAIA)."
    run_line = "**Run:** `python main.py all_divisions`"
    title = "# All_Divisions External Validation Findings"
    art_path = artifact_subdir or ""
    artifacts_header = (
        f"## Artifacts (`output/All_Divisions/{art_path}/`)"
        if art_path
        else "## Artifacts (`output/All_Divisions/`)"
    )

    if min_results_from > 1 and min_results_to > 1 and min_results_from == min_results_to:
        filter_line = (
            f"**Filter:** ≥{min_results_from} results in **both** from-event (A) and "
            f"to-event (B) within the athlete-season before predicting A→B. "
            "Same All_Divisions men's scrape (NCAA D1 / D3 / NAIA)."
        )
        run_line = "**Run:** `python main.py all_divisions_2plus`"
        title = "# All_Divisions · 2+_Results External Validation Findings"
    elif min_results_from > 1 and min_results_to <= 1:
        filter_line = (
            f"**Filter:** ≥{min_results_from} results in from-event (A) only; "
            "no minimum on to-event (B). Same All_Divisions men's scrape "
            "(NCAA D1 / D3 / NAIA)."
        )
        run_line = "**Run:** `python main.py all_divisions_2plus_from`"
        title = "# All_Divisions · 2+_From_Only External Validation Findings"
    elif min_results_from > 1 or min_results_to > 1:
        filter_line = (
            f"**Filter:** ≥{min_results_from} results in A and ≥{min_results_to} in B "
            "(athlete-season). Same All_Divisions men's scrape."
        )
        title = f"# All_Divisions · A≥{min_results_from} B≥{min_results_to} Findings"

    lines = [
        title,
        "",
        filter_line,
        "**Protocol:** season-PB only (chronological omitted).",
        "**Tolerances (target event):** 100m 0.15s · 200m 0.30s · 400m 0.80s · 800m 1.50s · "
        "1500m 3.0s · 3000m 6.0s · 3000m SC 8.0s · 5000m 10.0s.",
        "**Models:** club-fit event-pair formulas, WA bands 750–950 / 800–1000 / 850–1050; "
        "**feature** route preferred.",
        run_line,
        "**Note:** `2026_Outdoor_Marks.pdf` is Air Force (pages 1–4 = men), not Colorado State.",
        "",
        "## Dataset",
        "",
        "| | |",
        "|--|--|",
        f"| Result rows (source) | **{len(df):,}** ({n_primary:,} primary dated + {n_suppl:,} supplementary season-PB) |",
        f"| Divisions | {', '.join(divs) if divs else 'n/a'} |",
        f"| Schools | {', '.join(colleges)} |",
        f"| Prediction rows (feature) | **{len(f):,}** |",
        f"| Athlete-seasons tested | **{n_ath}** |",
    ]
    if min_results_from > 1 or min_results_to > 1:
        lines.append(
            f"| Min results | A ≥**{min_results_from}**; B ≥**{min_results_to}** |"
        )
    lines += [
        "",
        "## Headline results (season-PB)",
        "",
        "| Route | *n* | MedAPE | Within tolerance |",
        "|-------|-----|--------|------------------|",
    ]
    if fr:
        lines.append(
            f"| **feature** | {fr['n']} | **{fr['medape']:.2f}%** | **{fr['tol']:.1%}** |"
        )
    if pr:
        lines.append(
            f"| pooled | {pr['n']} | {pr['medape']:.2f}% | {pr['tol']:.1%} |"
        )

    # Contrast vs unrestricted and both-sides 2+ when this is A-only
    if min_results_from > 1 and min_results_to <= 1:
        lines += [
            "",
            "### Contrast with other filters (feature route)",
            "",
            "| Filter | *n* | MedAPE | Within tol | Athlete-seasons |",
            "|--------|-----|--------|------------|-----------------|",
        ]
        # unrestricted
        un_ov = OUT_DIR / "model_validation" / "summary_overall.csv"
        both_ov = OUT_DIR / "2+_Results" / "model_validation" / "summary_overall.csv"
        for label, path in (
            ("Unrestricted (A≥1, B≥1)", un_ov),
            ("A≥2 only (this file)", overall),
            ("A≥2 and B≥2", both_ov),
        ):
            if not path.exists():
                continue
            ov_c = pd.read_csv(path)
            sub = ov_c[(ov_c["mode"] == "season_pb") & (ov_c["route_kind"] == "feature")]
            if sub.empty:
                continue
            r = sub.iloc[0]
            # athlete-seasons from predictions if available
            n_as = "—"
            pred_p = path.parent / "predictions.csv"
            if pred_p.exists():
                pp = pd.read_csv(pred_p)
                ff = pp[(pp["mode"] == "season_pb") & (pp["route_kind"] == "feature")]
                if not ff.empty:
                    n_as = str(ff.groupby(["college", "athlete", "season_year"]).ngroups)
            lines.append(
                f"| {label} | {int(r['n'])} | {float(r['median_abs_pct_error']):.2f}% | "
                f"{float(r['within_tol_rate']):.1%} | {n_as} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Prediction success definition",
        "",
        "For each athlete with season PBs in events A and B, and each applicable WA band:",
        "",
    ]
    if min_results_from > 1 and min_results_to > 1:
        lines += [
            f"0. Require ≥{min_results_from} recorded results in A **and** ≥{min_results_to} "
            "in B that season (otherwise skip the pair).",
            "",
        ]
    elif min_results_from > 1 and min_results_to <= 1:
        lines += [
            f"0. Require ≥{min_results_from} recorded results in A that season; "
            "B may have a single result (otherwise skip the pair).",
            "",
        ]
    lines += [
        "1. Predict B from A using the band’s time model (feature route when available).",
        "2. Compare predicted B to actual season PB in B.",
        "3. **Successful prediction** if |pred − actual| ≤ starter tolerance for event B.",
        "",
        "**Athlete success** for a pair: the athlete has **at least one** successful "
        "prediction for that pair (any band counts).",
        "",
        "## Highest success rates by WA band (prediction-level)",
        "",
    ]

    for band in ["750-950", "800-1000", "850-1050"]:
        sub = by_band[(by_band["band"] == band) & (by_band["n"] >= 5)].sort_values(
            ["success_rate", "n"], ascending=[False, False]
        ) if not by_band.empty else pd.DataFrame()
        band_feat = f[f["band"] == band] if not f.empty else f
        overall_rate = float(band_feat["within_tolerance"].mean()) if len(band_feat) else float("nan")
        lines.append(f"### Band {band} (overall {overall_rate:.1%})" if len(band_feat) else f"### Band {band}")
        lines.append("")
        lines.append("| Pair | *n* | Successes | Rate | MedAPE |")
        lines.append("|------|-----|-----------|------|--------|")
        for _, r in sub.head(5).iterrows():
            lines.append(
                f"| {r['from_event']} → {r['to_event']} | {int(r['n'])} | "
                f"{int(r['successes'])} | **{r['success_rate']:.1%}** | {r['medape']:.2f}% |"
            )
        lines.append("")

    lines += [
        "## Highest athlete success rates by event pair",
        "",
        "| Pair | Athletes | Successful | Athlete success rate | Row success rate | MedAPE |",
        "|------|----------|------------|----------------------|------------------|--------|",
    ]
    if not by_ath.empty:
        for _, r in by_ath.head(12).iterrows():
            lines.append(
                f"| {r['from_event']} → {r['to_event']} | {int(r['n_athletes'])} | "
                f"{int(r['n_successful_athletes'])} | **{r['athlete_success_rate']:.1%}** | "
                f"{r['row_success_rate']:.1%} | {r['medape']:.2f}% |"
            )

    if not f.empty:
        ath_any = f.groupby(["college", "athlete", "season_year"])["within_tolerance"].any()
        lines += [
            "",
            f"**Overall athlete-level (≥1 hit on any pair):** {int(ath_any.sum())} / {len(ath_any)} = "
            f"**{ath_any.mean():.1%}** (feature), vs row-level within-tol "
            f"**{f['within_tolerance'].mean():.1%}**.",
        ]

    lines += [
        "",
        artifacts_header,
        "",
    ]
    if min_results_from <= 1 and min_results_to <= 1:
        lines += [
            "- `men_test_dataset_results.csv`",
            "- `men_athlete_season_features.csv`",
            "- `men_supplementary_season_pb.csv`",
            "- `README.txt`",
        ]
    else:
        lines += [
            "- Source scrape reused from parent `All_Divisions/men_test_dataset_results.csv`",
        ]
    lines += [
        "- `model_validation/predictions.csv`",
        "- `model_validation/summary_overall.csv`",
        "- `model_validation/summary_by_band.csv`",
        "- `model_validation/summary_by_pair.csv`",
        "- `model_validation/validation_report.txt`",
        "- `model_validation/success_rate_by_band_pair.csv`",
        "- `model_validation/athlete_success_rate_by_pair.csv`",
        "- `model_validation/research/` (inferential CSVs, report, plots)",
        "",
    ]

    path = root / findings_filename
    path.write_text("\n".join(lines) + "\n")
    print(f"[{tag}] Wrote {path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="All_Divisions time-model validation pipeline"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=(
            "preprocess",
            "validate",
            "validate_2plus",
            "validate_2plus_from",
            "all",
            "2plus",
            "2plus_from",
        ),
    )
    args = parser.parse_args(argv)
    if args.command in ("preprocess", "all"):
        run_preprocess_all_divisions()
    if args.command in ("validate", "all"):
        run_validate_all_divisions()
    if args.command in ("validate_2plus", "2plus"):
        # Reuse existing scrape; only re-validate with 2+ filter on A and B
        run_validate_all_divisions(
            artifact_subdir="2+_Results",
            min_results_from=2,
            min_results_to=2,
        )
    if args.command in ("validate_2plus_from", "2plus_from"):
        # ≥2 in A only; B unrestricted (≥1)
        run_validate_all_divisions(
            artifact_subdir="2+_From_Only",
            min_results_from=2,
            min_results_to=1,
            findings_filename="Summary_findings.md",
        )


if __name__ == "__main__":
    main()
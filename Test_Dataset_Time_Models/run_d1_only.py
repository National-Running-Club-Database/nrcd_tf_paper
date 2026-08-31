"""D1_Only preprocess + validate + research pipeline.

Uses only NCAA D1 schools (existing D1 scrapers + newly added D1 documents).
Keeps the same absolute-second tolerance bands as the main validation.
All artifacts are prefixed with D1_Only.
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
    OUT_DIR,
    add_features,
    athlete_feature_summary,
    load_coefficients,
    scrape_depaul,
    scrape_north_florida,
    scrape_siue,
    scrape_usi_supplementary,
)

PREFIX = "D1_Only"


def run_preprocess_d1_only() -> Path:
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
        *NEW_D1_SUPPLEMENTARY,
    ]

    all_rows: list[dict] = []
    counts: dict[str, int] = {}
    for label, fn in primary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"[D1_Only] Scraped {label}: {len(part)} dated results")

    for label, fn in supplementary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"[D1_Only] Scraped {label}: {len(part)} season-best rows")

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise SystemExit("D1_Only: no rows scraped")

    # Hard filter — only NCAA D1
    raw = raw[raw["College_Division"] == "NCAA D1"].copy()
    before = len(raw)
    raw = raw.drop_duplicates(
        subset=["College", "Athlete", "Event", "Result", "Competition_Date", "Meet"]
    )
    print(f"[D1_Only] Deduped {before - len(raw)} rows; {len(raw)} remain (NCAA D1 only)")

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

    results_path = OUT_DIR / f"{PREFIX}_men_test_dataset_results.csv"
    features_path = OUT_DIR / f"{PREFIX}_men_athlete_season_features.csv"
    suppl_path = OUT_DIR / f"{PREFIX}_men_supplementary_season_pb.csv"
    readme_path = OUT_DIR / f"{PREFIX}_README.txt"

    featured.to_csv(results_path, index=False)
    summary.to_csv(features_path, index=False)
    suppl = featured[featured["Source_Role"] == "supplementary_season_pb"]
    suppl.to_csv(suppl_path, index=False)

    n_primary = int((featured["Source_Role"] == "primary_dated").sum())
    n_suppl = int((featured["Source_Role"] == "supplementary_season_pb").sum())
    colleges = sorted(featured["College"].dropna().unique())

    lines = [
        "D1_Only Men's Test Dataset — Time Models",
        "========================================",
        "",
        "Filter: College_Division == NCAA D1 only.",
        "Tolerances: same starter absolute-second / metre bands as main pipeline.",
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
        f"Total rows: {len(featured)} (primary_dated={n_primary}, supplementary_season_pb={n_suppl})",
        f"Athletes×season: {featured.groupby(['College', 'Athlete', 'Season_Year']).ngroups}",
        "",
        f"Outputs: {results_path.name}, {features_path.name}, {suppl_path.name}",
    ]
    readme_path.write_text("\n".join(lines) + "\n")
    print(f"[D1_Only] Wrote {results_path}")
    print(f"[D1_Only] Wrote {features_path}")
    print(f"[D1_Only] Wrote {suppl_path}")
    print(f"[D1_Only] Wrote {readme_path}")
    return results_path


def run_validate_d1_only() -> None:
    """Season-PB validation only on NCAA D1 men's data."""
    import validate as V

    test_csv = OUT_DIR / f"{PREFIX}_men_test_dataset_results.csv"
    if not test_csv.exists():
        raise SystemExit(f"Missing {test_csv}; run preprocess first")

    out_dir = OUT_DIR / "model_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    old_test, old_out = V.TEST_CSV, V.OUT_DIR
    V.TEST_CSV = test_csv
    V.OUT_DIR = out_dir

    try:
        df = pd.read_csv(test_csv)
        # NCAA D1 + men only
        if "College_Division" in df.columns:
            df = df[df["College_Division"] == "NCAA D1"].copy()
        if "Gender" in df.columns:
            df = df[df["Gender"].astype(str).str.lower().isin(["men", "m", "male"])].copy()
        # Drop any residual mislabeled non-D1 college strings
        df = df[df["College"].astype(str).str.lower() != "colorado state"].copy()

        print(
            f"[D1_Only] Validation frame: {len(df)} rows | "
            f"colleges={sorted(df['College'].dropna().unique())}"
        )

        pb = V.load_test_pb(df)

        all_models: list = []
        for band, fname in V.BAND_FILES.items():
            path = V.MODELS_DIR / fname
            parsed = V.parse_band_file(path, band)
            print(f"[D1_Only] Parsed {band}: {len(parsed)} Men pair models")
            all_models.extend(parsed)

        # Season-PB protocol only (user request)
        pred = V.evaluate(all_models, df, pb, "season_pb")
        overall, by_band, by_pair = V.summarize(pred)

        pred_path = out_dir / f"{PREFIX}_predictions.csv"
        overall_path = out_dir / f"{PREFIX}_summary_overall.csv"
        band_path = out_dir / f"{PREFIX}_summary_by_band.csv"
        pair_path = out_dir / f"{PREFIX}_summary_by_pair.csv"
        pred.to_csv(pred_path, index=False)
        overall.to_csv(overall_path, index=False)
        by_band.to_csv(band_path, index=False)
        by_pair.to_csv(pair_path, index=False)

        report_path = out_dir / f"{PREFIX}_validation_report.txt"
        tmp = V.write_report(pred, overall, by_band, by_pair, len(all_models))
        report_path.write_text(
            "D1_Only — NCAA Division I MEN only · SEASON-PB protocol only\n"
            "Same absolute tolerances as main validation pipeline.\n"
            "Chronological evaluation intentionally omitted.\n"
            "Note: 2026_Outdoor_Marks.pdf is Air Force (GOAIRFORCEFALCONS.COM), "
            "not Colorado State; men pages 1–4 only.\n\n"
            + tmp.read_text()
        )

        print(f"[D1_Only] Wrote {pred_path} ({len(pred)} rows)")
        print(f"[D1_Only] Wrote {overall_path}")
        print(f"[D1_Only] Wrote {band_path}")
        print(f"[D1_Only] Wrote {pair_path}")
        print(f"[D1_Only] Wrote {report_path}")
        print()
        print(report_path.read_text()[:2800])

        print("\n--- D1_Only research statistics (season_pb) ---\n")
        _run_research_d1_only(pred)
    finally:
        V.TEST_CSV = old_test
        V.OUT_DIR = old_out


def _run_research_d1_only(pred: pd.DataFrame) -> None:
    import research_stats as R

    research_dir = OUT_DIR / "model_validation" / f"{PREFIX}_research"
    plot_dir = research_dir / "plots"
    research_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    old_pred, old_out, old_plot = R.PRED_PATH, R.OUT_DIR, R.PLOT_DIR
    # Point research_stats at a temp predictions path it expects, or call internals
    tmp_pred = OUT_DIR / "model_validation" / f"{PREFIX}_predictions.csv"
    R.PRED_PATH = tmp_pred
    R.OUT_DIR = research_dir
    R.PLOT_DIR = plot_dir
    try:
        R.run_research_stats(pred)
        # Prefix key research artifacts that research_stats writes without prefix
        for name in [
            "inferential_overall.csv",
            "inferential_by_pair.csv",
            "feature_vs_pooled_tests.csv",
            "inferential_report.txt",
        ]:
            src = research_dir / name
            if src.exists():
                dst = research_dir / f"{PREFIX}_{name}"
                dst.write_bytes(src.read_bytes())
                print(f"[D1_Only] Wrote {dst}")
    finally:
        R.PRED_PATH = old_pred
        R.OUT_DIR = old_out
        R.PLOT_DIR = old_plot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="D1_Only time-model validation pipeline")
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("preprocess", "validate", "all"),
    )
    args = parser.parse_args(argv)
    if args.command in ("preprocess", "all"):
        run_preprocess_d1_only()
    if args.command in ("validate", "all"):
        run_validate_d1_only()


if __name__ == "__main__":
    main()

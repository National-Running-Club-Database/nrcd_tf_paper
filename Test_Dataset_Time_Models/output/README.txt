Men's Test Dataset — Time Models
================================

Primary sources (per-result competition dates; season-PB + chronological):
  • SIUE outdoor performance list → 117 rows
  • North Florida 2024 outdoor results → 101 rows
  • DePaul 2026 outdoor results (TF_2025_Results.pdf, Men's pages only) → 201 rows

Supplementary sources (undated; season-best rows only for season-PB validation):
  • USI outdoor performance list → 76 rows
  • North Central outdoor HTML matrix → 243 rows
  • Wisconsin-Oshkosh outdoor HTML matrix → 146 rows
  • Keiser outdoor HTML matrix → 58 rows
  • DePaul Women's pages inside TF_2025_Results.pdf (men folder run) — skipped

Design note:
  Season-PB transfer is the stronger external-validation protocol for these
  club-fit models. Supplementary undated lists expand that protocol. Chronological
  early→later checks use Source_Role=primary_dated rows only.

Total result rows: 942  (primary_dated=419, supplementary_season_pb=523)
Athlete-seasons: 320
Athlete-season-groups with bal_spec: 371  ({'balanced': 216, 'specialized': 155})

World Athletics scores:
  Approx. outdoor 2025 quadratic coefficients from
  https://github.com/jchen1/iaaf-scoring-tables (coefficients-2025.json).

Feature definitions:
  • bal_spec: balanced if WA_Spread < 50; else specialized
  • best_event: event with highest season-best World_Athletics_Score_Men in group
  • Source_Role: primary_dated | supplementary_season_pb
  • Tolerance: starter absolute error band for predicted vs actual

Outputs:
  • men_test_dataset_results.csv
  • men_athlete_season_features.csv
  • men_supplementary_season_pb.csv

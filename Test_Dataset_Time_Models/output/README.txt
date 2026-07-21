Men's Test Dataset — Time Models
================================

Sources used (have competition dates):
  • SIUE outdoor performance list → 117 rows
  • North Florida 2024 outdoor results → 101 rows
  • DePaul 2026 outdoor results (TF_2025_Results.pdf, Men's pages only) → 201 rows

Sources skipped (no per-result competition dates):
  • 2025-26 USI Men's Track & Field Performance List - University of Southern Indiana Athletics.pdf
  • 2025-26 Men's Outdoor Track & Field Statistics - North Central College Athletics.html
  • 2026 Men's Outdoor Track & Field Statistics - University of Wisconsin-Oshkosh Athletics.html
  • 2026 MOTF Stats - Keiser University Athletics.html
  • DePaul Women's pages inside TF_2025_Results.pdf (Men folder run)

Total result rows: 419
Athlete-seasons: 77
Athlete-season-groups with bal_spec: 83  ({'balanced': 55, 'specialized': 28})

World Athletics scores:
  Approx. outdoor 2025 quadratic coefficients from
  https://github.com/jchen1/iaaf-scoring-tables (coefficients-2025.json).
  Men's dataset still records World_Athletics_Score_Women for the same mark
  (useful for cross-checks; primary routing uses Men's scores).

Feature definitions:
  • bal_spec: balanced if WA_Spread < 50; else specialized
    WA_Spread = max(event-best WA) − min(event-best WA) within Event_Group
  • best_event: event with highest season-best World_Athletics_Score_Men in group
  • Result_Order_In_Season / Season_Day_Index: chronological markers for
    early-season → later-season prediction checks
  • Tolerance: starter absolute error band for predicted vs actual

Outputs:
  • men_test_dataset_results.csv
  • men_athlete_season_features.csv

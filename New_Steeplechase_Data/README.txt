New_Steeplechase_Data — Distance RQ1 / RQ1B / RQ1C
=================================================

Purpose
-------
Re-run Distance research questions with updated 3000m Steeplechase World Athletics
point scoring, mirroring Relays_Findings outputs:

  Distance_Relays_Findings/   → RQ1 best-event (relay-inclusive)
  RQ1B_Nationals/             → nationals top-8 WA distributions + 8th-place thresholds
  RQ1C_Competitiveness/       → which event is most competitive vs nationals bar

How to re-run
-------------
  python3 run_distance_rq_analyses.py

Or step-by-step:
  python3 prepare_distance_data.py
  python3 analyze_rq1_best_event_distance.py
  python3 analyze_rq1b_nationals_distance.py
  python3 analyze_rq1c_competitiveness_distance.py

Data note
---------
See DATA_NOTES.txt. After the gender/scoring fix, all six Relays_Distance_*
CSVs pass gender validation and use the corrected steeplechase WA points
(means roughly doubled vs the legacy Relays_Findings copies).

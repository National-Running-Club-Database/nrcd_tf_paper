First-event recommendations — multiple metrics
==============================================

Band: outdoor first-event World Athletics Points ∈ [750, 950).
Same dual indoor+outdoor athlete-year cohort for all metrics.

Files:
  first_event_recommendations_band_750_950_wa.txt       (copied)
  first_event_recommendations_band_750_950_vdot.txt
  first_event_recommendations_band_750_950_purdy.txt
  first_event_recommendations_band_750_950_mercier.txt
  first_event_band_750_950_detail_{wa,vdot,purdy,mercier}.csv

Regenerate non-WA metrics:
  python indoor_analysis/Indoor_Outdoor_Interplay/analyze_first_event_multi_metric.py

Season-opener indoor vs outdoor (band 750–950)
----------------------------------------------
  opener_findings_band_750_950_wa.txt            (copied)
  opener_findings_band_750_950_vdot.txt
  opener_findings_band_750_950_purdy.txt
  opener_findings_band_750_950_mercier.txt
  opener_pair_comparisons_band_750_950_{metric}.csv
  opener_summary_slices_band_750_950_{metric}.csv

  Regenerate opener non-WA metrics:
    python indoor_analysis/Indoor_Outdoor_Interplay/analyze_opener_multi_metric.py

Outdoor-only first-event recommendations (band 750–950)
-------------------------------------------------------
  outdoor_first_event_recommendations_band_750_950_{wa,vdot,purdy,mercier}.txt
  outdoor_first_event_band_750_950_detail_{metric}.csv

  Same dual-season cohort as first_event_recommendations_*; indoor sections omitted.
  Regenerate:
    python indoor_analysis/Indoor_Outdoor_Interplay/analyze_outdoor_first_event_multi_metric.py

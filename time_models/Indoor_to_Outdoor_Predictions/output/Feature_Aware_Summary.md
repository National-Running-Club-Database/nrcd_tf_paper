# Feature-Aware Indoor → Outdoor Predictions

Reports in the style of `feature_aware_indoor_to_outdoor_findings.txt`.

## Unbanded

- [`feature_aware_indoor_to_outdoor_findings.txt`](feature_aware_indoor_to_outdoor_findings.txt)

## By indoor from-event WA band

Band = WA of the athlete's **indoor season-best** (fastest/best mark) in the from-event.

| Band | Report | Models CSV |
|------|--------|------------|
| 750–950 | [`feature_aware_indoor_to_outdoor_band_750_950.txt`](feature_aware_indoor_to_outdoor_band_750_950.txt) | [`by_indoor_wa_band/indoor_to_outdoor_models_band_750_950.csv`](by_indoor_wa_band/indoor_to_outdoor_models_band_750_950.csv) |
| 800–1000 | [`feature_aware_indoor_to_outdoor_band_800_1000.txt`](feature_aware_indoor_to_outdoor_band_800_1000.txt) | [`by_indoor_wa_band/indoor_to_outdoor_models_band_800_1000.csv`](by_indoor_wa_band/indoor_to_outdoor_models_band_800_1000.csv) |
| 850–1050 | [`feature_aware_indoor_to_outdoor_band_850_1050.txt`](feature_aware_indoor_to_outdoor_band_850_1050.txt) | [`by_indoor_wa_band/indoor_to_outdoor_models_band_850_1050.csv`](by_indoor_wa_band/indoor_to_outdoor_models_band_850_1050.csv) |

```bash
python time_models/Indoor_to_Outdoor_Predictions/main.py feature
```


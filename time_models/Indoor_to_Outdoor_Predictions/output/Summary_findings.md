# Indoor → Outdoor Predictions — Summary Findings

Linear models predict outdoor season-best from indoor season-best for the requested event pairs (2024–2026, same athlete-year).

## Models by gender

### Men

| Pair | n | r | MedAE | MAPE | Formula |
|------|---|---|-------|------|---------|
| Indoor 60m → Outdoor 100m | 389 | 0.817 | 0.198 | 2.23% | `Outdoor 100m = 0.3034 + 1.5339 × Indoor 60m  (s)` |
| Indoor 200m → Outdoor 200m | 361 | 0.856 | 0.425 | 2.22% | `Outdoor 200m = 1.4392 + 0.9286 × Indoor 200m  (s)` |
| Indoor 400m → Outdoor 400m | 264 | 0.910 | 0.924 | 2.32% | `Outdoor 400m = 0.9985 × Indoor 400m − 0.3532  (s)` |
| Indoor 800m → Outdoor 800m | 410 | 0.858 | 2.776 | 2.90% | `Outdoor 800m = 28.5862 + 0.7763 × Indoor 800m  (s)` |
| Indoor Mile → Outdoor 1500m | 662 | 0.913 | 4.961 | 2.52% | `Outdoor 1500m = 19.1158 + 0.8580 × Indoor Mile  (s)` |
| Indoor 3000m → Outdoor 5000m | 312 | 0.919 | 16.407 | 2.10% | `Outdoor 5000m = 89.0023 + 1.5729 × Indoor 3000m  (s)` |
| Indoor 3000m → Outdoor 1500m | 289 | 0.843 | 5.751 | 2.82% | `Outdoor 1500m = 52.0753 + 0.3684 × Indoor 3000m  (s)` |
| Indoor 3000m → Outdoor 3000m Steeplechase | 86 | 0.809 | 19.335 | 3.92% | `Outdoor 3000m Steeplechase = 66.1961 + 1.0255 × Indoor 3000m  (s)` |
| Indoor 5000m → Outdoor 5000m | 126 | 0.913 | 14.910 | 2.18% | `Outdoor 5000m = 109.5127 + 0.8863 × Indoor 5000m  (s)` |
| Indoor Long Jump → Outdoor Long Jump | 114 | 0.869 | 0.196 | 4.28% | `Outdoor Long Jump = 0.4542 + 0.9327 × Indoor Long Jump  (m)` |
| Indoor Triple Jump → Outdoor Triple Jump | 54 | 0.826 | 0.294 | 3.15% | `Outdoor Triple Jump = 1.5314 + 0.8914 × Indoor Triple Jump  (m)` |
| Indoor High Jump → Outdoor High Jump | 52 | 0.799 | 0.047 | 3.30% | `Outdoor High Jump = 0.3644 + 0.7938 × Indoor High Jump  (m)` |
| Indoor 60m Hurdles → Outdoor 110m Hurdles | 85 | 0.812 | 0.505 | 4.32% | `Outdoor 110m Hurdles = 1.0316 + 1.7377 × Indoor 60m Hurdles  (s)` |
| Indoor Shot Put → Outdoor Shot Put | 113 | 0.928 | 0.459 | 5.60% | `Outdoor Shot Put = 0.0113 + 0.9970 × Indoor Shot Put  (m)` |

### Women

| Pair | n | r | MedAE | MAPE | Formula |
|------|---|---|-------|------|---------|
| Indoor 60m → Outdoor 100m | 171 | 0.915 | 0.212 | 1.90% | `Outdoor 100m = 1.7440 × Indoor 60m − 1.2906  (s)` |
| Indoor 200m → Outdoor 200m | 196 | 0.925 | 0.404 | 1.93% | `Outdoor 200m = 0.2269 + 0.9761 × Indoor 200m  (s)` |
| Indoor 400m → Outdoor 400m | 120 | 0.953 | 1.205 | 2.25% | `Outdoor 400m = 2.6873 + 0.9449 × Indoor 400m  (s)` |
| Indoor 800m → Outdoor 800m | 185 | 0.927 | 3.276 | 2.62% | `Outdoor 800m = 1.0148 × Indoor 800m − 3.2924  (s)` |
| Indoor Mile → Outdoor 1500m | 292 | 0.929 | 6.372 | 2.59% | `Outdoor 1500m = 21.4416 + 0.8602 × Indoor Mile  (s)` |
| Indoor 3000m → Outdoor 5000m | 112 | 0.916 | 24.519 | 2.28% | `Outdoor 5000m = 1.7604 × Indoor 3000m − 34.0510  (s)` |
| Indoor 3000m → Outdoor 1500m | 98 | 0.926 | 5.929 | 2.53% | `Outdoor 1500m = 0.4640 × Indoor 3000m − 3.1616  (s)` |
| Indoor 3000m → Outdoor 3000m Steeplechase | 39 | 0.604 | 36.160 | 5.88% | `Outdoor 3000m Steeplechase = 248.9228 + 0.7804 × Indoor 3000m  (s)` |
| Indoor 5000m → Outdoor 5000m | 44 | 0.925 | 16.613 | 2.14% | `Outdoor 5000m = 19.6643 + 0.9713 × Indoor 5000m  (s)` |
| Indoor Long Jump → Outdoor Long Jump | 87 | 0.840 | 0.196 | 4.58% | `Outdoor Long Jump = 1.1528 + 0.7739 × Indoor Long Jump  (m)` |
| Indoor Triple Jump → Outdoor Triple Jump | 44 | 0.933 | 0.200 | 2.75% | `Outdoor Triple Jump = 0.2973 + 0.9879 × Indoor Triple Jump  (m)` |
| Indoor High Jump → Outdoor High Jump | 46 | 0.759 | 0.034 | 3.19% | `Outdoor High Jump = 0.2521 + 0.8230 × Indoor High Jump  (m)` |
| Indoor 60m Hurdles → Outdoor 100m Hurdles | 70 | 0.844 | 0.492 | 3.33% | `Outdoor 100m Hurdles = 4.0895 + 1.2991 × Indoor 60m Hurdles  (s)` |
| Indoor Shot Put → Outdoor Shot Put | 73 | 0.847 | 0.411 | 6.13% | `Outdoor Shot Put = 1.6080 + 0.8199 × Indoor Shot Put  (m)` |

## Artifacts

- `output/indoor_to_outdoor_models.csv`
- `output/indoor_to_outdoor_predictions.csv`
- `output/indoor_to_outdoor_summary.csv`
- `output/indoor_to_outdoor_findings.txt`


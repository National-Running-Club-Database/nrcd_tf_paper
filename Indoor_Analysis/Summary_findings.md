# Summary Findings — indoor_analysis

**Data:** NIRCA club indoor track 2024–2026 (no outdoor March-1 filter; all dates in file).
**Scope:** RQ1A specialization only — no indoor Nationals exists, so RQ1B/RQ1C are not run.
**Full reports:** [RQ1A_Best_Event/rq1a_indoor_summary.txt](RQ1A_Best_Event/rq1a_indoor_summary.txt),
[Point_Bands_Time_Models/point_band_findings_width_200.txt](Point_Bands_Time_Models/point_band_findings_width_200.txt),
[Indoor_Outdoor_Interplay/indoor_outdoor_opener_wa_findings_band_750_950.txt](Indoor_Outdoor_Interplay/indoor_outdoor_opener_wa_findings_band_750_950.txt)

---

## 1. Bottom line

Indoor specialization patterns broadly mirror outdoor (short sprints and the mile/800m
dominate best-event counts), with one notable **gender asymmetry in distance**: women
who double into longer indoor distances tend to score higher there, while men's shorter
event (800m) beats the mile head-to-head. Athletes with both indoor and outdoor seasons
consistently **open outdoor meets with higher WA scores than their indoor openers**
(+27.0 WA mean, +24.0 median, 69.3% of comparisons favor outdoor).

## 2. RQ1A — indoor best-event specialization

**Sprints** (athletes with ≥2 of 60m/200m/400m):

| Gender | Modal best event | 2nd | 3rd |
|--------|-------------------|-----|-----|
| Men | 60m (274) | 200m (181) | 400m (96) |
| Women | 60m (112) | 400m (50) | 200m (33) |

Pairwise: men's 60m beats 200m (319–207) and 400m (122–102); 200m beats 400m (171–128).
Women's 60m dominates 200m (161–37) but is roughly even with 400m (39–36); 400m beats
200m (89–49) — the same 400m-over-200m pattern seen in outdoor women's sprints.

**Distance** (athletes with ≥2 of 800m/Mile/3000m/5000m):

| Gender | Modal best event | 2nd | 3rd | 4th |
|--------|-------------------|-----|-----|-----|
| Men | Mile (242) | 800m (215) | 3000m (194) | 5000m (51) |
| Women | Mile / 3000m tie (101 each) | 800m (94) | 5000m (36) | — |

Pairwise: men's 800m beats Mile (260–205); Mile ≈ 3000m (192–189, roughly even);
3000m beats 5000m (90–48). **Women show a clear gender asymmetry**: longer events win
for dual athletes — 3000m beats Mile (110–43) and 800m (34–20); 5000m beats Mile (30–16)
and 800m (14–5) — i.e., women who double into longer indoor distances tend to score
higher there, echoing the outdoor women's 1500m-vs-5000m pattern.

**Hurdles, Jumps, Throws**: indoor field/hurdle events have small dual-event samples
(e.g., 55m vs. 60m Hurdles: men n=2, women n=0; Shot Put/Discus dual throwers: men n=5,
women n=2) — RQ1A specialization is **not well identified** indoors for these disciplines.
Jumps show men favoring Long Jump (34) then High Jump (24) then Triple Jump (19); women
favor Triple Jump (21) then Long Jump (18) then High Jump (14), with High Jump strongly
beating Triple Jump for men (12–3, small n).

## 3. Indoor point-band time models

Testing WA point bands of width 200 (starting every 50 points) for Sprints
(60/200/400) and Distance (800/Mile/3000):

| Band | Mean CV error | Pairs | Athlete-seasons |
|------|---------------|-------|------------------|
| **800–1000** | **2.129s** | 16 | 319 |
| 750–950 | 2.919s | 20 | — |
| 850–1050 | 2.987s | 8 | — |
| 700–900 | 3.038s | 22 | — |
| 650–850 | 3.291s | 22 | — |
| Unbanded (all) | 5.012s | 24 | — |

The 800–1000 band is both the most accurate **and** meets the ≥12-pair stability
threshold — banding cuts mean CV error by more than half versus no banding at all
(2.129s vs. 5.012s).

## 4. Indoor → outdoor season-opener transition

Among athletes with ≥2 indoor and ≥2 outdoor meets in the same year, restricted to
comparisons where the outdoor opener WA falls in [750, 950):

- **Combined (2024–2026): n=218 comparisons, 161 athletes** — indoor opener mean WA
  789.3, outdoor opener mean WA 816.3 (**Δ = +27.0**, median +24.0); outdoor opener
  higher in **69.3%** of comparisons, indoor higher in 29.4%.
- **By year:** the outdoor advantage was largest in 2025 (Δ = +35.0, 75.9% outdoor-higher)
  and smallest in 2026 (Δ = +14.4, 60.0% outdoor-higher).
- **By event pair:** Long Jump→Long Jump (Δ = +55.6) and Mile→1500m (Δ = +49.8) show the
  largest gains; 800m→800m is nearly a coin flip (46.2% outdoor-higher, Δmedian = **-2.0**).
- **Athlete-level** (mean delta per athlete-year across their comparable pairs):
  n = 184 athlete-years, mean Δ = +27.02, outdoor higher 69.6% of the time.

**Verdict (from the source report):** outdoor season openers average higher WA than
indoor season openers for comparable events among dual-season athletes.

## 5. Limitations

1. **No indoor Nationals** — RQ1B (nationals top-8) and RQ1C (competitiveness vs.
   nationals bar) cannot be run for indoor track; this folder only answers RQ1A.
2. **Relay-inflated "all athletes" counts** — indoor discipline folders include shared
   relay rows; specialization reads should use the `_2plus_individual` / pairwise
   `_individual` outputs, not the raw "all athletes" counts, for Hurdles/Jumps/Throws.
3. **No March-1 outdoor-style date filter** — indoor season timing conventions differ
   from outdoor, so all dates in the source files are used as-is.
4. **Selection into opener comparisons is not random** — an athlete must run the paired
   event at *both* season openers to be included (e.g., many milers open outdoor in the
   800m and would be excluded from the Mile→1500m comparison).
5. **Cross-event WA approximations** — 60m→100m, Mile→1500m, and 3000m→5000m compare
   different race distances on the WA scoring scale by design; treat as approximate.
6. **Small samples for several event pairs** — several event pairs in the opener
   comparison have n < 10 (e.g., 5000m→5000m n=5, Shot Put→Shot Put n=4); point estimates
   there are illustrative, not robust.

## 6. Reproducibility

```bash
cd indoor_analysis
python main.py all
```

See [README.md](README.md) for the subfolder scripts run separately and folder dependencies.

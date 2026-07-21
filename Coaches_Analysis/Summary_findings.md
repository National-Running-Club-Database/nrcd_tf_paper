# Summary Findings — coaches_analysis

**Data:** RQ1B 8th-place WA thresholds + RQ1C nationals-margin comparisons, relay-inclusive,
NIRCA outdoor 2024–2026 (March 1+).
**Full reports:** [Nationals_Scoring_Focus_Men_Women.txt](Nationals_Scoring_Focus_Men_Women.txt),
[Nationals_Scoring_Focus_Clearing_Roster.txt](Nationals_Scoring_Focus_Clearing_Roster.txt),
[Nationals_Points_Maximization_Clearing_Roster.txt](Nationals_Points_Maximization_Clearing_Roster.txt)

---

## 1. Bottom line

Three coaching frames answer three different questions from the same RQ1B/RQ1C data:
**clear-rate** (which events are most accessible for a roster still building toward
nationals?), **margin+spread** (which events give clearing-capable athletes the biggest
buffer while avoiding internal competition?), and **points-maximization** (which thin,
high-margin events should you *stack* talent into for total team points?). The
recommended primary event **changes** depending on which frame you use — e.g., men's
sprints go from **4x100m** (clear-rate frame) to **400m** (points-stacking frame).

## 2. Frame 1 — Clear-rate / scorer-count (accessible bars)

**Men's team, ranked scoring portfolio:** 4x100m (133 clearers, 26.3%) → 110m Hurdles
(38 clearers, 32.5% clear rate — best in men's track) → 200m (72 clearers) → High Jump
(29 clearers, 27.9%) → 4x400m (depth only, 9.9%) → 1500m/Steeple/Discus (specialist paths).
**Avoid steering toward:** 800m (829 WA bar, only 4.6% clear), open 100m, long jump,
treating 4x400m as a default "best event."

**Women's team, ranked scoring portfolio:** 4x100m (105 clearers, 37.2% — best conversion
in the dataset) → 400m Hurdles (32 clearers, 40.0% clear rate) → High Jump (28 clearers,
37.3%) → 1500m (46 clearers) → Discus (25 clearers, 29.1%) → 4x400m (18.0% clear, more
viable than for men) → Steeplechase/100mH/Triple Jump (niche plays).
**Avoid steering toward:** open 100m/200m/400m (7–9% clear rates), 5000m unless
specialist, long jump as default, javelin/hammer (0% clear rate for both genders).

**Key gender differences:** men's best hurdle event is 110mH (32.5% clear) vs. women's
400mH (40.0%); both genders favor High Jump for jumps; men's hardest bar is 800m (829
WA) vs. women's 100m/5000m (814/809 WA); distance volume response is stronger for women
(~+17 WA/race) than men (~+12 WA/race).

## 3. Frame 2 — Margin + field density (clearing-capable rosters)

Once many athletes already clear the bar, clear-rate stops mattering and **margin above
the bar + national field size** drives placement:

| Group | Men — margin-favored event | Women — margin-favored event |
|-------|------------------------------|----------------------------------|
| Sprints | 400m (smallest open-sprint pool: 42 clearers) | 400m (smallest pool: 23 clearers) |
| Distance | Steeplechase (dominates every margin comparison, 24 clearers) | Steeplechase (75–6 margin vs. 1500m, 22 clearers) |
| Hurdles | 400m Hurdles (beats 110mH 42–27 on margin) | 400m Hurdles (beats 100mH 26–18, beats 4x400 25–9) |
| Jumps | Triple Jump (beats Long 61–19, beats High 21–11) | High Jump (beats Long 24–15, beats Triple 12–6) |
| Throws | Shot Put (beats Discus 67–45 on margin) | Discus (beats Shot 41–32 on margin) |

**4x100m remains worth staffing (one A team per gender)** but is deprioritized as a
*depth* strategy because it has the largest national clearing pool (133 men / 105 women)
— margin is strong there, but the field is deepest, so additional relay depth returns
less than spreading talent into thinner individual events.

## 4. Frame 3 — Points maximization (stack, don't spread)

With standard 10-8-6-5-4-3-2-1 scoring, four athletes placing 1st–4th in one event (29
points) beats four athletes placing 6th–8th in a crowded event (9 points) or one relay
win (10 points, but uses 4 athletes for a single scoring line). This flips the
recommendation from Frame 1/2's "spread talent" logic to **"stack thin, high-margin events."**

**Highest point-potential stacks:**

| Rank | Men | Women |
|------|-----|-------|
| 1 | 3000m Steeplechase (24 clearers, dominant margins) | 3000m Steeplechase (22 clearers, dominant margins) |
| 2 | Triple Jump (23 clearers) | 400m (23 clearers) |
| 3 | 400m (42 clearers) | 400m Hurdles (32 clearers, 40% clear rate) |
| 4 | Shot Put (22 clearers) | High Jump (28 clearers) |
| 5 | 5000m (36 clearers) | Discus (25 clearers) |
| 6 | 110m Hurdles (38 clearers) | 5000m (29 clearers) |

**Events to avoid stacking** (crowded national pools, low marginal point value):
Men — 4x100m (133), 4x400m (81), 200m (72), 1500m (51), 800m (46). Women — 4x100m (105),
4x400m (78), 1500m (46). **Relay rule:** staff exactly one A relay per gender (usually
4x100m) for its own scoring line, but never at the expense of individual stacks.

## 5. Cross-cutting season-structure notes

- Men's distance and women's distance/hurdles show the strongest volume→improvement
  signal (+12 to +22 WA per additional race); men's hurdles, jumps, and throws show
  weak or no volume benefit — quality/timing matters more than race count there.
- Roughly half of multi-meet athletes across groups still peak in the final 25% of
  their schedule — season-structure advice ("build volume early, sharpen late") is
  the one constant across all three coaching frames; **event choice** changes across
  frames, but **timing advice does not**.

## 6. Limitations

1. **Only 4x100m and 4x400m have valid relay WA scoring** — all relay-related
   recommendations are limited to these two relay types (see `relays_findings/Summary_findings.md`).
2. **Descriptive synthesis, not new analysis** — these reports re-package RQ1B/RQ1C
   numbers into decision frames; they inherit every limitation of the underlying RQ1
   analyses (WA points are comparability scores, NIRCA club-only sample, selection bias
   in event entry — see `non_relays_findings/Summary_findings.md` §6).
3. **Scoring assumption**: the points-maximization frame assumes standard
   10-8-6-5-4-3-2-1 championship scoring; a different scoring table would change the
   stack-vs-spread math.
4. **Static roster assumption** — none of the three frames models roster changes across
   a season (injuries, new events, athlete development); recommendations are best used
   as a starting point for a season plan, not a fixed rule.
5. **Distance/steeple numbers here use the original (uncorrected) steeplechase scoring**
   — see `new_steeplechase_data/coaches_analysis/README.txt` for how the corrected
   scoring shifts the steeplechase bar (~456 → ~740 WA for men) while leaving the
   overall recommendation (steeple is still the top distance margin play) unchanged.

## 7. Reproducibility

```bash
cd coaches_analysis
python main.py
```

No computation is performed — see [README.md](README.md) for the source data chain.

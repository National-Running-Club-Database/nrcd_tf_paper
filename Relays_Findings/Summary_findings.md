# Summary Findings — relays_findings (Relay-Inclusive RQ1)

**Data:** NIRCA club outdoor track 2024–2026, individual events **plus** 4x100m/4x400m
relays (only relay types with nonzero WA scoring); team WA credited to every leg athlete.
**Full reports:** [relay_analysis_trends_summary.txt](relay_analysis_trends_summary.txt),
[relay_inclusion_rq_differences.txt](relay_inclusion_rq_differences.txt)

---

## 1. Bottom line

Including relays **fundamentally changes RQ1A and RQ1C but leaves RQ1B's individual-event
benchmarks untouched**. The 4x400m becomes the modal "best event" and modal "most
competitive event" across nearly every discipline (because team WA scores are often
higher than individual marks), while **4x100m — not 4x400m — actually has the highest
nationals clear-rates**. Distance is the one discipline where individual events (1500m,
5000m) largely hold their ground.

## 2. RQ1A — best event flips to 4x400m

| Group | Without relays (top event, n) | With relays (top event, n) |
|-------|-------------------------------|------------------------------|
| Men sprinters | 100m (465) | **4x400m (611)**, then 100m (351), 4x100m (308) |
| Women sprinters | 100m (245) | **4x400m (322)**, then 4x100m (216), 100m (142) |
| Men hurdles | 400m Hurdles (154) | **4x400m (688)**, then 4x100m (395) |
| Men jumps | Long Jump (198) | **4x400m (699)**, then 4x100m (372) |
| Men throws | Shot Put (127) | **4x400m (707)**, then 4x100m (394) |
| Men distance | 1500m (816) | **1500m still #1 (723)**, then 800m (487), 4x400m (475) |

Distance is the exception: men's 1500m (723) and 800m (487) still outrank 4x400m (475).
Across all individual×relay pairwise comparisons, relays win **~48 of 63 pair types**
vs. ~15 for individual events; hurdles favor relays in **0 of 8** pair types, while
distance favors individuals in **6 of 16**.

## 3. RQ1B — relay thresholds slot in mid-pack (individual thresholds unchanged)

3-year average 8th-place WA thresholds:

| Gender | Highest individual | 4x400m | 4x100m |
|--------|--------------------|--------|--------|
| Men | 800m 829.3, 1500m 827.0 | 785.3 | 760.3 |
| Women | 100m 814.3, 5000m 808.5 | 747.7 | 776.0 |

4x100m is notably more accessible than the open 100m for both genders (men 818.0 →
760.3; women 814.3 → 776.0). Individual-event thresholds are **unchanged** by relay
inclusion — RQ1B answers for individual events remain directly comparable to
`non_relays_findings/`.

## 4. RQ1C — the 4x100m vs. 4x400m paradox

| Metric | Men | Women |
|--------|-----|-------|
| Modal RQ1C event (sprints) | **4x400m** (614 athletes), 9.9% clear bar | **4x400m** (345 athletes), 18.0% clear bar |
| Highest population clear-rate | **4x100m**: 133/506 (26.3%) | **4x100m**: 105/282 (37.2%) |

Many athletes have their *largest margin* above threshold at 4x400m (because its bar is
low), but far fewer actually clear it in absolute terms than at 4x100m. **4x100m is the
better actionable relay target**; 4x400m simply looks better on the margin metric.

**RQ1A vs. RQ1C agreement** (same athlete, same discipline, relay-inclusive): highest for
men's throws (91.5%), women's sprints (91.0%), men's sprints (89.8%); lowest for women's
distance (75.8%), women's throws (76.5%), women's hurdles (77.3%). Distance shows the
most RQ1A/RQ1C disagreement even with relays included.

**Individual vs. relay head-to-head margins** (dual-event athletes): relays usually win —
men 100m vs. 4x100m: 4x100m wins 209 of 258; women 100m vs. 4x100m: 4x100m wins 124 of
136. Distance retains more individual strength: men 5000m vs. 4x400m is a near-split
(81–76); men 3000m SC vs. 4x400m slightly favors steeple (46–40).

## 5. What changes vs. the individual-only analysis (RQ4 sensitivity check)

| RQ | Effect of adding relays |
|----|--------------------------|
| RQ1A | **Largest change** — 4x400m becomes modal best event in nearly every discipline except distance; individual best-event counts fall sharply (e.g., men's 100m 465 → 351). |
| RQ1B | **Additive only** — two new relay benchmarks; all individual thresholds unchanged. |
| RQ1C | **Major reframing** — relays (especially 4x400m) become the modal recommendation in sprints/hurdles/jumps/throws; population clear-counts are led by 4x100m. |

Discipline athlete counts also grow substantially once relay legs are counted (e.g., men's
hurdles 238 → 1,228; men's jumps 318 → 1,325; men's throws 191 → 1,285) — this reflects
**roster-wide relay participation**, not a change in event specialization for those athletes.

**Hypotheses (from the paper framework):** H1C-1 (RQ1A≠RQ1C) still holds with relays,
though the size of the mismatch shifts by discipline/gender (e.g., men's jumps agreement
rises +12.4 pp with relays; men's throws falls −4.9 pp). H1C-2 (women lean 5000m over
1500m) and H1C-3 (steeple's conditional competitive advantage) are **unchanged** by relay
inclusion — steeple still beats 1500m on margin among dual-event athletes (132–39 men;
75–6 women), even though steeple loses RQ1C recommendation *share* to relays.

## 6. Limitations

1. **Only 4x100m and 4x400m have WA scoring** — 4x200m, 4x800m, DMR, SMR, and Swedish
   Relay appear in the raw data but are excluded from all threshold-based comparisons
   because nationals 8th-place WA scores are missing or zero. This is a data gap, not a
   methodological choice.
2. **Team WA credited to every leg** inflates "best event"/"most competitive event"
   labels for athletes whose relay teammates carry the mark; this is explicit in the
   design (per-leg credit) but should not be read as "this athlete individually ran that fast."
3. **Roster crossover** — hurdle/jump/throw athletes who leg relays inflate 4x400m/4x100m
   counts in disciplines where they aren't specialists; interpret discipline-level RQ1C
   relay dominance as roster opportunity, not a signal that specialists should switch events.
4. Same general RQ1 limitations as `non_relays_findings/` apply: WA points are
   comparability scores, NIRCA club-only sample, selection bias in event entry, and
   inconsistent nationals round labeling.

## 7. Reproducibility

```bash
cd relays_findings
python main.py all
```

See [README.md](README.md) for per-discipline scripts and folder dependencies.

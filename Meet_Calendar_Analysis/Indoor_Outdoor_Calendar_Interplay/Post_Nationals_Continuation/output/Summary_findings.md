# Post-Nationals Continuation — Summary Findings

**Question:** If athletes keep racing after April NIRCA Nationals, do indoor↔outdoor calendar recommendations change?

**Share with ≥1 post-nats outdoor meet:** 33.8% of outdoor athlete-seasons. Policy cohort: **815** multi-event continuers with indoor.

## Short answer

**Mostly no — the logic stays the same; outdoor *volume* targets rise slightly.**

- Indoor still = diagnose + build PBs (≈2–4 meets).
- Pre-nats outdoor still = lock RQ1C (≈2–3 meets).
- Post-nats adds ~**+1 meet / +2–4 races** for distance & sprints; less for jumps/throws/hurdles.
- Extra outdoor slots favor **more championship-event volume**, not a new explore-first strategy.

## Runway shift (continuers vs stoppers)

| Group | n | Mean outdoor meets (full) | Mean outdoor races (full) |
|-------|---|---------------------------|---------------------------|
| Continuers | 1,893 | 1.80 | 2.35 |
| Stop at/before nats | 3,704 | 1.23 | 1.66 |

## Policy ranking still prefers lock/volume over explore

See `post_nats_calendar_findings.txt` for full Ni/No tables. At extended No=3–4, **greedy** and **Indoor RQ1A → outdoor RQ1C** remain top practical policies; explore-indoors stays behind specialize.

## Artifacts

- `post_nats_calendar_findings.txt` (full write-up)
- `post_nats_policy_comparison.csv`
- `post_nats_runway_by_gender_discipline.csv`

## Run

```bash
python Meet_Calendar_Analysis/Indoor_Outdoor_Calendar_Interplay/Post_Nationals_Continuation/analyze_post_nats.py
```


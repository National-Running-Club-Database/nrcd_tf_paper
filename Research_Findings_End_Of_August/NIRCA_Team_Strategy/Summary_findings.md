# Indoor ↔ Outdoor Calendar Interplay — Summary Findings

**Question:** How to structure indoor + early outdoor calendars together for April NIRCA nationals margin?

**Cohort:** 1,335 multi-event athlete-seasons with both indoor and pre-nats outdoor results (of 1,823 with both phases).

## Why combine the calendars?

Outdoor pre-nationals racing is scarce (often **1 meet**). Indoor (Dec–Mar) holds most volume (typically **1–2+ meets**). Exploration belongs indoors; outdoor slots should lock the championship event.

## Empirical runway (both phases)

- Mean indoor meets: **1.89** (median 2)
- Mean outdoor meets before nationals: **1.30** (median 1)
- Mean combined meets: **3.19**

Most common joint budgets:

| Indoor meets | Outdoor meets | Athletes |
|--------------|---------------|----------|
| 1 | 1 | 670 |
| 2 | 1 | 381 |
| 3 | 1 | 187 |
| 1 | 2 | 185 |
| 2 | 2 | 130 |
| 3 | 2 | 64 |
| 4 | 1 | 54 |
| 4 | 2 | 38 |

## Continuity: indoor RQ1C → outdoor focus

- Outdoor focus matches indoor RQ1C: **57.9%**
- Outdoor absolute-best is indoor RQ1C: **56.1%**
- Mean outdoor margin if continuous: **-201.6** vs switchers **-193.0** (Δ **-8.6 WA**)
- Interpretation: switchers are **not** worse — outdoor can still correct indoor rankings; use continuity as the default, with optional one-meet correction.

## Policy comparison (mean nationals margin)

### Indoor slots Ni=2 · Outdoor slots No=1

| Policy | n | Mean margin | Clear rate |
|--------|---|-------------|------------|
| Hindsight oracle (upper bound) | 1335 | -119.8 | 28.4% |
| Indoor RQ1A → outdoor RQ1C | 1335 | -157.3 | 18.6% |
| Greedy margin both phases | 1335 | -158.8 | 19.2% |
| Observed indoor+outdoor choices | 1335 | -167.6 | 17.0% |
| Take indoor as-is → specialize outdoor RQ1C | 1335 | -168.2 | 17.1% |
| Specialize RQ1C both phases | 1335 | -168.9 | 16.4% |
| Indoor transfer-partner → outdoor RQ1C | 1335 | -178.3 | 14.5% |
| Explore indoors → lock RQ1C outdoors | 1335 | -183.1 | 13.8% |

### Indoor slots Ni=2 · Outdoor slots No=2

| Policy | n | Mean margin | Clear rate |
|--------|---|-------------|------------|
| Hindsight oracle (upper bound) | 1335 | -101.3 | 34.4% |
| Greedy margin both phases | 1335 | -142.3 | 22.7% |
| Indoor RQ1A → outdoor RQ1C | 1335 | -143.6 | 21.6% |
| Specialize RQ1C both phases | 1335 | -152.1 | 20.4% |
| Take indoor as-is → specialize outdoor RQ1C | 1335 | -153.2 | 20.6% |
| Observed indoor+outdoor choices | 1335 | -158.5 | 19.0% |
| Indoor transfer-partner → outdoor RQ1C | 1335 | -161.9 | 18.7% |
| Explore indoors → lock RQ1C outdoors | 1335 | -167.0 | 17.8% |

### Indoor slots Ni=3 · Outdoor slots No=2

| Policy | n | Mean margin | Clear rate |
|--------|---|-------------|------------|
| Hindsight oracle (upper bound) | 1335 | -83.3 | 37.3% |
| Greedy margin both phases | 1335 | -128.9 | 26.1% |
| Indoor RQ1A → outdoor RQ1C | 1335 | -129.8 | 24.9% |
| Specialize RQ1C both phases | 1335 | -141.3 | 23.0% |
| Take indoor as-is → specialize outdoor RQ1C | 1335 | -148.4 | 21.4% |
| Indoor transfer-partner → outdoor RQ1C | 1335 | -152.4 | 20.4% |
| Observed indoor+outdoor choices | 1335 | -154.7 | 19.7% |
| Explore indoors → lock RQ1C outdoors | 1335 | -156.6 | 19.3% |

## Coaching answer

1. **Indoor = build the PB vector.** Race the events that define absolute best and nationals margin; bank marks (proxies OK: mile→1500, 3000→5000).
2. **Outdoor = lock the current RQ1C.** With No=1–2, specialize; do not casually re-shop.
3. **Do not burn indoor slots on blind exploration** when Ni≈2 — greedy / RQ1A→RQ1C beats explore-then-lock.
4. **Allow one outdoor correction** if a new event clearly flips the margin ranking (~42% switch in the data; switchers are not worse on average).
5. **Count relays** against both indoor and outdoor meet budgets.

Best practical policy at Ni=2, No=2 in simulation: **Greedy margin both phases**.

## Artifacts

- `output/interplay_findings.txt`
- `output/combined_policy_comparison.csv`
- `output/indoor_outdoor_continuity.csv`
- `output/joint_indoor_outdoor_meets.csv`
- `output/combined_runway_distribution.csv`

## Run

```bash
python Meet_Calendar_Analysis/Indoor_Outdoor_Calendar_Interplay/analyze_interplay.py
```


# Meet Calendar Analysis — Summary Findings

**Question:** Given N outdoor meets before April NIRCA nationals, which events maximize end-of-season nationals margin?

**Nationals dates:** 2024-04-06 · 2025-04-05 · 2026-04-11 (indoor season + typically **2–3 outdoor meets** beforehand).

**Cohort:** 2,211 multi-event athlete-seasons; volume lift used for counterfactual repeats = **18.0 WA**.

## Outdoor runway (empirical)

- Pre-nationals athlete-seasons: **4,373**
- With indoor prior: **41.7%**

| Outdoor meets before nationals | Athletes | Share |
|--------------------------------|----------|-------|
| 1 | 3,357 | 76.8% |
| 2 | 912 | 20.9% |
| 3 | 99 | 2.3% |
| 4 | 4 | 0.1% |
| 5 | 1 | 0.0% |

## Policy comparison (mean nationals margin, WA points)

### N = 2

| Policy | n | Mean margin | Median margin | Clear rate |
|--------|---|-------------|---------------|------------|
| Hindsight greedy oracle (upper bound) | 2211 | -156.9 | -132.0 | 22.6% |
| Greedy expected-margin each slot | 2211 | -171.5 | -146.3 | 19.0% |
| Specialize on RQ1C (nationals-margin event) | 2211 | -173.9 | -152.0 | 18.6% |
| Specialize on RQ1A (absolute best WA) | 2211 | -174.4 | -149.3 | 18.5% |
| Observed chronological choices | 2211 | -186.8 | -163.7 | 16.4% |
| Explore alternate meet 1 → lock RQ1C | 2211 | -199.0 | -174.0 | 15.2% |

### N = 3

| Policy | n | Mean margin | Median margin | Clear rate |
|--------|---|-------------|---------------|------------|
| Hindsight greedy oracle (upper bound) | 2211 | -138.9 | -114.0 | 26.8% |
| Greedy expected-margin each slot | 2211 | -153.5 | -128.3 | 22.5% |
| Specialize on RQ1A (absolute best WA) | 2211 | -156.6 | -131.3 | 21.8% |
| Specialize on RQ1C (nationals-margin event) | 2211 | -157.8 | -134.5 | 21.6% |
| Observed chronological choices | 2211 | -181.0 | -156.3 | 17.3% |
| Explore alternate meet 1 → lock RQ1C | 2211 | -182.2 | -157.7 | 18.4% |

## Coaching answer

With only **2–3 outdoor meets** before April nationals:

1. **Decide the championship event indoors** (RQ1C = closest to the nationals bar).
2. **Default: specialize** — spend outdoor slots on that event. Greedy margin allocation
   and RQ1C specialization both beat the observed mixed calendars (~+13 to +27 WA).
3. **Do not burn meet 1 on exploration by default** — indoor RQ1A=RQ1C for ~94% of
   athletes with priors; explore-then-lock *hurts* mean margin when N is only 2–3.
4. **Explore only when priors disagree** (RQ1A≠RQ1C, ~6%): then meet 1 = RQ1C candidate,
   meets 2–N lock the winner.
5. Time-model doubles (100↔200, 800→1500) are for information, not substitutes for
   championship volume. Count relays against N.

Best practical policies in simulation (excl. oracle/observed):

- N=2: **Greedy expected-margin each slot** (mean margin -171.5)
- N=3: **Greedy expected-margin each slot** (mean margin -153.5)

**Indoor RQ1A≠RQ1C rate:** 5.6% of athletes with indoor priors — these are the athletes who most need Template B.

## Artifacts

- `output/meet_calendar_findings.txt`
- `output/policy_comparison_overall.csv`
- `output/policy_comparison.csv`
- `output/pre_nationals_meet_distribution.csv`
- `output/recommendations_by_profile.csv`
- `output/athlete_policy_detail.csv`

## Run

```bash
python Meet_Calendar_Analysis/main.py
# or: python main.py run Meet_Calendar_Analysis
```


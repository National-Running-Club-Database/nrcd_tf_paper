# Cursor AI Track Paper

Reproducible analyses for a *Journal of Quantitative Analysis in Sports* manuscript on **event specialization in collegiate club track & field** using World Athletics (WA) scoring and the National Running Club Database (NRCD).

Companion XC work: *Faster Results From A Smarter Schedule* (PDF in this repo).

## Quick start

```bash
# create / use project venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# list modules
python main.py

# run one module
python main.py run test_dataset_time_models
python main.py run causal_analysis research
python main.py run non_relays_findings all
```

Each analysis folder has the same interface:

```text
<module>/
  main.py              # CLI entry (default: all)
  README.md            # methods + how to run
  Summary_findings.md  # results narrative with real numbers
```

## Analysis modules (pipeline order)

| Module | Role |
|--------|------|
| [non_relays_findings](non_relays_findings/) | Outdoor RQ1A/B/C + RQ3 (individual events) |
| [relays_findings](relays_findings/) | Same RQs with relays; shared data helpers for time models |
| [new_steeplechase_data](new_steeplechase_data/) | **Canonical** distance/steeple WA re-run + FI models |
| [indoor_analysis](indoor_analysis/) | Indoor RQ1A, point-band/FI models, indoor↔outdoor interplay |
| [number_of_events_question](number_of_events_question/) | Competition volume vs seasonal WA point jump |
| [causal_analysis](causal_analysis/) | Within-athlete fixed-effects dose–response |
| [time_models](time_models/) | Cross-event time models, point bands, feature importance |
| [test_dataset_time_models](test_dataset_time_models/) | External NCAA D1 validation of pair models |
| [coaches_analysis](coaches_analysis/) | Coach-facing nationals strategy narratives |
| [preliminary_findings](preliminary_findings/) | Curated presentation copies (not runnable) |

Master question plan: [Research_Questions.md](Research_Questions.md)  
Project-wide results synthesis: [Summary_findings.md](Summary_findings.md)

## Research standards used in this repo

Across quantitative modules we emphasize:

- **Uncertainty** — bootstrap / cluster-bootstrap 95% CIs; Clopper–Pearson for rates where used
- **Hypothesis tests** — Wilcoxon, Spearman, paired comparisons (not point estimates alone)
- **Commensurate metrics** — percent error for cross-event time models; WA points for specialization
- **Design honesty** — FE / dose–response framed as within-athlete association, not RCT causal proof
- **External validation** — club-fit models tested on dated NCAA D1 marks (`test_dataset_time_models`)

Template module (snake_case layout + full inferential suite): `test_dataset_time_models/`.

## Naming note

Top-level analysis folders use **snake_case** (`relays_findings/`, `causal_analysis/`, etc.). Nested discipline folders such as `Sprinters_Relays_Findings/` retain their historical names for now. Obsolete Feature Importance “copy” folders were removed; canonical FI code lives under `time_models/Point_Bands_Time_Models/Feature_Importance_Point_Band_Time_Models/` and steeple-aware copies under `new_steeplechase_data/`.

## Dependencies

See [requirements.txt](requirements.txt). Primary stack: pandas, numpy, scipy, matplotlib, pdfplumber.

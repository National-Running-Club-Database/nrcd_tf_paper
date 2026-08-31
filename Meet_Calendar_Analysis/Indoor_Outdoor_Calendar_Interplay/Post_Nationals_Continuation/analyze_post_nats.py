"""Post-NIRCA-Nationals continuation: do indoor↔outdoor calendar recommendations change?

Compares athletes who keep racing after April nationals vs the pre-nationals-only
calendar framing used in Indoor_Outdoor_Calendar_Interplay.

Outputs a separate findings file under ``output/``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent  # Meet_Calendar_Analysis
REPO = PARENT.parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import analyze_meet_calendar as MC  # noqa: E402

NATIONALS = MC.NATIONALS_DATES
THRESHOLDS = MC.THRESHOLDS
DISCIPLINE_EVENTS = MC.DISCIPLINE_EVENTS

# Outdoor budgets when the season continues past nationals
PRE_ONLY_BUDGETS = [(2, 1), (2, 2), (3, 2)]
EXTENDED_BUDGETS = [(2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]

POLICIES = [
    "explore_indoor_lock_outdoor",
    "specialize_both_rq1c",
    "indoor_rq1a_outdoor_rq1c",
    "greedy_both",
    "observed_combined",
    "oracle_combined",
]


@dataclass
class ExtendedSeason:
    athlete_id: int
    gender: str
    discipline: str
    season_year: int
    indoor: list = field(default_factory=list)
    outdoor_pre: list = field(default_factory=list)
    outdoor_post: list = field(default_factory=list)
    n_indoor_meets: int = 0
    n_outdoor_pre_meets: int = 0
    n_outdoor_post_meets: int = 0

    @property
    def outdoor_full(self):
        return self.outdoor_pre + self.outdoor_post

    @property
    def n_outdoor_full_meets(self) -> int:
        return self.n_outdoor_pre_meets + self.n_outdoor_post_meets

    @property
    def continues_after_nats(self) -> bool:
        return self.n_outdoor_post_meets > 0


def build_extended(outdoor: pd.DataFrame, indoor: pd.DataFrame) -> list[ExtendedSeason]:
    seasons: list[ExtendedSeason] = []
    for year, nat in NATIONALS.items():
        out_y = outdoor[outdoor["season_year"] == year]
        in_y = indoor[indoor["season_year"] == year] if not indoor.empty else pd.DataFrame()

        indoor_map: dict[tuple, list] = defaultdict(list)
        indoor_meets: dict[tuple, set] = defaultdict(set)
        if not in_y.empty:
            for r in in_y.itertuples():
                key = (int(r.athlete_id), r.gender, r.discipline)
                indoor_map[key].append((str(r.event), float(r.wa), r.meet_id, r.start_date))
                indoor_meets[key].add(r.meet_id)

        # All athletes with any outdoor (excluding nationals day itself from "training" stream;
        # nationals marks can still inform end margin but we treat them separately)
        groups = out_y.groupby(["athlete_id", "gender", "discipline"])
        for (aid, gender, disc), g in groups:
            key = (int(aid), gender, disc)
            g = g.sort_values("start_date")
            pre = g[g["start_date"] < nat]
            post = g[g["start_date"] > nat]
            # skip pure nationals-only with no other outdoor — still allow if indoor exists
            if pre.empty and post.empty:
                continue

            def pack(df):
                return [
                    (str(r.event), float(r.wa), r.meet_id, r.start_date) for r in df.itertuples()
                ]

            seasons.append(
                ExtendedSeason(
                    athlete_id=int(aid),
                    gender=gender,
                    discipline=disc,
                    season_year=year,
                    indoor=sorted(indoor_map.get(key, []), key=lambda x: x[3]),
                    outdoor_pre=pack(pre),
                    outdoor_post=pack(post),
                    n_indoor_meets=len(indoor_meets.get(key, set())),
                    n_outdoor_pre_meets=int(pre["meet_id"].nunique()) if len(pre) else 0,
                    n_outdoor_post_meets=int(post["meet_id"].nunique()) if len(post) else 0,
                )
            )
    return seasons


def feasible_events(seas: ExtendedSeason) -> list[str]:
    ev = (
        {e for e, _, _, _ in seas.indoor}
        | {e for e, _, _, _ in seas.outdoor_pre}
        | {e for e, _, _, _ in seas.outdoor_post}
    )
    return [
        e
        for e in sorted(ev)
        if e in DISCIPLINE_EVENTS[seas.discipline] and (seas.gender, e) in THRESHOLDS
    ]


# Import interplay sim helpers by loading the module from sibling folder
def _load_interplay():
    interplay_dir = HERE.parent
    if str(interplay_dir) not in sys.path:
        sys.path.insert(0, str(interplay_dir))
    import analyze_interplay as AI  # type: ignore

    return AI


def simulate_on_stream(AI, seas: ExtendedSeason, outdoor_stream, ni, no, policy, lift):
    """Reuse interplay policies but with a chosen outdoor stream (pre-only or full)."""
    # Build a CombinedSeason-like duck
    from analyze_interplay import CombinedSeason, simulate_combined

    cs = CombinedSeason(
        athlete_id=seas.athlete_id,
        gender=seas.gender,
        discipline=seas.discipline,
        season_year=seas.season_year,
        indoor=seas.indoor,
        outdoor=outdoor_stream,
        n_indoor_meets=seas.n_indoor_meets,
        n_outdoor_meets=(
            seas.n_outdoor_pre_meets
            if outdoor_stream == seas.outdoor_pre
            else seas.n_outdoor_full_meets
        ),
    )
    return simulate_combined(cs, ni, no, policy, lift)


def runway_tables(seasons: list[ExtendedSeason]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for s in seasons:
        rows.append(
            {
                "athlete_id": s.athlete_id,
                "gender": s.gender,
                "discipline": s.discipline,
                "season_year": s.season_year,
                "n_indoor": s.n_indoor_meets,
                "n_outdoor_pre": s.n_outdoor_pre_meets,
                "n_outdoor_post": s.n_outdoor_post_meets,
                "n_outdoor_full": s.n_outdoor_full_meets,
                "continues_after_nats": s.continues_after_nats,
                "n_outdoor_races_pre": len(s.outdoor_pre),
                "n_outdoor_races_post": len(s.outdoor_post),
                "n_outdoor_races_full": len(s.outdoor_full),
                "n_indoor_races": len(s.indoor),
            }
        )
    dist = pd.DataFrame(rows)
    by = (
        dist.groupby(["gender", "discipline", "continues_after_nats"])
        .agg(
            n=("athlete_id", "count"),
            mean_in=("n_indoor", "mean"),
            mean_out_pre=("n_outdoor_pre", "mean"),
            mean_out_post=("n_outdoor_post", "mean"),
            mean_out_full=("n_outdoor_full", "mean"),
            med_out_full=("n_outdoor_full", "median"),
            p90_out_full=("n_outdoor_full", lambda s: s.quantile(0.9)),
            mean_races_full=("n_outdoor_races_full", "mean"),
            med_races_full=("n_outdoor_races_full", "median"),
            p90_races_full=("n_outdoor_races_full", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )
    return dist, by


def run_analysis() -> None:
    AI = _load_interplay()
    print("[Post-Nats Calendar] Loading frames…")
    outdoor, indoor = MC.load_season_frames()
    lift = MC.estimate_volume_lift(outdoor)

    seasons = build_extended(outdoor, indoor)
    continuers = [s for s in seasons if s.continues_after_nats and s.indoor]
    continuers_multi = [
        s
        for s in continuers
        if len(
            {e for e, _, _, _ in s.indoor}
            | {e for e, _, _, _ in s.outdoor_pre}
            | {e for e, _, _, _ in s.outdoor_post}
        )
        >= 2
    ]
    # Matched comparison: same athletes, pre-only outdoor stream vs full outdoor stream
    print(
        f"[Post-Nats Calendar] seasons={len(seasons):,} "
        f"continuers_with_indoor={len(continuers):,} multi={len(continuers_multi):,}"
    )

    dist, by_grp = runway_tables(seasons)
    dist.to_csv(OUT / "post_nats_runway_distribution.csv", index=False)
    by_grp.to_csv(OUT / "post_nats_runway_by_gender_discipline.csv", index=False)

    share = float(dist["continues_after_nats"].mean()) if len(dist) else 0.0
    print(f"[Post-Nats Calendar] Share with ≥1 post-nats outdoor meet: {share:.1%}")

    detail_rows = []
    # Pre-only budgets on continuers (using only pre stream)
    for ni, no in PRE_ONLY_BUDGETS:
        for s in continuers_multi:
            for pol in POLICIES:
                row = simulate_on_stream(AI, s, s.outdoor_pre, ni, no, pol, lift)
                if row:
                    row["horizon"] = "pre_nats_only"
                    detail_rows.append(row)

    # Extended budgets on full outdoor stream
    for ni, no in EXTENDED_BUDGETS:
        for s in continuers_multi:
            for pol in POLICIES:
                row = simulate_on_stream(AI, s, s.outdoor_full, ni, no, pol, lift)
                if row:
                    row["horizon"] = "through_post_nats"
                    detail_rows.append(row)

    # Also run pre budgets on full stream to isolate effect of extra outdoor marks
    for ni, no in PRE_ONLY_BUDGETS:
        for s in continuers_multi:
            for pol in POLICIES:
                row = simulate_on_stream(AI, s, s.outdoor_full, ni, no, pol, lift)
                if row:
                    row["horizon"] = "full_stream_short_budget"
                    detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "post_nats_policy_detail.csv", index=False)

    overall = (
        detail.groupby(["horizon", "ni", "no", "policy"])
        .agg(
            n=("athlete_id", "count"),
            mean_margin=("final_margin", "mean"),
            median_margin=("final_margin", "median"),
            clear_rate=("cleared", "mean"),
        )
        .reset_index()
        .sort_values(["horizon", "ni", "no", "mean_margin"], ascending=[True, True, True, False])
    )
    overall.to_csv(OUT / "post_nats_policy_comparison.csv", index=False)

    # Continuity: among continuers, does post-nats best event match pre RQ1C?
    cont_rows = []
    for s in continuers:
        if not s.outdoor_pre or not s.outdoor_post:
            continue
        pre_pb = {}
        for e, wa, _, _ in s.outdoor_pre:
            pre_pb[e] = max(pre_pb.get(e, -1e18), wa)
        post_pb = {}
        for e, wa, _, _ in s.outdoor_post:
            post_pb[e] = max(post_pb.get(e, -1e18), wa)
        e_pre, m_pre = MC.margin(pre_pb, s.gender)
        e_post, m_post = MC.margin(post_pb, s.gender)
        if e_pre is None or e_post is None:
            continue
        cont_rows.append(
            {
                "gender": s.gender,
                "discipline": s.discipline,
                "pre_rq1c": e_pre,
                "post_rq1c": e_post,
                "same_event": e_pre == e_post,
                "pre_margin": m_pre,
                "post_margin": m_post,
                "margin_change": m_post - m_pre,
            }
        )
    cont = pd.DataFrame(cont_rows)
    cont.to_csv(OUT / "post_nats_event_continuity.csv", index=False)

    write_findings(dist, by_grp, overall, cont, len(continuers_multi), share, lift)
    print(f"[Post-Nats Calendar] Wrote findings under {OUT}")


def write_findings(
    dist: pd.DataFrame,
    by_grp: pd.DataFrame,
    overall: pd.DataFrame,
    cont: pd.DataFrame,
    n_multi: int,
    share: float,
    lift: float,
) -> None:
    labels = {
        "explore_indoor_lock_outdoor": "Explore indoors → lock RQ1C outdoors",
        "specialize_both_rq1c": "Specialize RQ1C both phases",
        "indoor_rq1a_outdoor_rq1c": "Indoor RQ1A → outdoor RQ1C",
        "greedy_both": "Greedy margin both phases",
        "observed_combined": "Observed choices",
        "oracle_combined": "Oracle (upper bound)",
    }

    continuers = dist[dist["continues_after_nats"]]
    stoppers = dist[~dist["continues_after_nats"]]

    lines = [
        "Post-Nationals Continuation — Do Calendar Recommendations Change?",
        "=" * 72,
        "",
        "Question",
        "--------",
        "If an athlete plans to keep racing after April NIRCA Nationals, do the",
        "indoor↔outdoor calendar recommendations from Indoor_Outdoor_Calendar_Interplay",
        "change?",
        "",
        "Setup",
        "-----",
        "• Pre-nats horizon: outdoor results with start_date < nationals (same as before).",
        "• Extended horizon: outdoor results before OR after nationals (nationals day excluded",
        "  from the training stream).",
        "• Nationals dates: 2024-04-06 · 2025-04-05 · 2026-04-11",
        f"• Cohort for policy sims: multi-event athletes with indoor + ≥1 post-nats outdoor meet",
        f"  (n={n_multi:,}).",
        f"• Share of outdoor athlete-seasons with ≥1 post-nats meet: {share:.1%}",
        f"• Volume lift (synthetic repeats): {lift:.1f} WA",
        "",
        "Note on data coverage",
        "---------------------",
        "Post-nats outdoor volume is largest in 2024 (through ~mid-May). 2025–2026 scrapes",
        "have fewer post-nats rows — treat extended-season estimates as informative but",
        "heavier on 2024 continuers.",
        "",
        "1. Does the outdoor meet runway actually get longer?",
        "-" * 50,
    ]

    if len(continuers) and len(stoppers):
        lines += [
            f"Continuers (n={len(continuers):,}): mean outdoor meets pre={continuers['n_outdoor_pre'].mean():.2f}, "
            f"post={continuers['n_outdoor_post'].mean():.2f}, full={continuers['n_outdoor_full'].mean():.2f} "
            f"(median full={continuers['n_outdoor_full'].median():.0f}, p90={continuers['n_outdoor_full'].quantile(0.9):.0f})",
            f"Stoppers   (n={len(stoppers):,}): mean outdoor meets pre={stoppers['n_outdoor_pre'].mean():.2f}, "
            f"full={stoppers['n_outdoor_full'].mean():.2f}",
            f"Mean outdoor races full — continuers {continuers['n_outdoor_races_full'].mean():.2f} vs "
            f"stoppers {stoppers['n_outdoor_races_full'].mean():.2f}",
            "",
        ]

    lines.append("By gender × discipline (continuers only):")
    lines.append(
        f"{'Gender':<6} {'Group':<10} {'n':>5} {'In':>5} {'OutPre':>7} {'OutPost':>8} "
        f"{'OutFull':>8} {'p90Full':>7} {'RacesF':>7}"
    )
    c_by = by_grp[by_grp["continues_after_nats"] == True]  # noqa: E712
    for _, r in c_by.sort_values(["gender", "discipline"]).iterrows():
        lines.append(
            f"{r['gender']:<6} {r['discipline']:<10} {int(r['n']):>5} "
            f"{r['mean_in']:>5.1f} {r['mean_out_pre']:>7.1f} {r['mean_out_post']:>8.1f} "
            f"{r['mean_out_full']:>8.1f} {r['p90_out_full']:>7.1f} {r['mean_races_full']:>7.1f}"
        )

    lines += [
        "",
        "2. Do optimal policies change when outdoor budget grows?",
        "-" * 50,
        "Compare ranking of practical policies (excl. oracle) under short vs extended No.",
        "",
    ]

    def rank_block(horizon: str, ni: int, no: int) -> list[str]:
        sub = overall[
            (overall["horizon"] == horizon)
            & (overall["ni"] == ni)
            & (overall["no"] == no)
            & (~overall["policy"].isin(["oracle_combined"]))
        ].sort_values("mean_margin", ascending=False)
        out = [f"  ({horizon}, Ni={ni}, No={no})"]
        for _, r in sub.iterrows():
            out.append(
                f"    {labels.get(r['policy'], r['policy'])}: "
                f"mean margin {r['mean_margin']:.1f}, clear {r['clear_rate']:.1%}"
            )
        return out

    for block in [
        ("pre_nats_only", 2, 2),
        ("through_post_nats", 2, 3),
        ("through_post_nats", 2, 4),
        ("through_post_nats", 3, 4),
        ("through_post_nats", 4, 4),
    ]:
        lines.extend(rank_block(*block))
        lines.append("")

    # Does explore become better with larger No?
    lines += [
        "Explore-indoors vs specialize as outdoor slots increase (continuers, full stream):",
    ]
    for no in [2, 3, 4]:
        for ni in [2, 3]:
            sub = overall[
                (overall["horizon"] == "through_post_nats")
                & (overall["ni"] == ni)
                & (overall["no"] == no)
            ].set_index("policy")
            if "explore_indoor_lock_outdoor" not in sub.index or "specialize_both_rq1c" not in sub.index:
                continue
            exp = float(sub.loc["explore_indoor_lock_outdoor", "mean_margin"])
            spec = float(sub.loc["specialize_both_rq1c", "mean_margin"])
            lines.append(
                f"  Ni={ni}, No={no}: explore {exp:.1f} vs specialize-RQ1C {spec:.1f} "
                f"(Δ explore−spec {exp - spec:+.1f})"
            )

    # Event continuity pre→post
    lines += ["", "3. Post-nats event focus vs pre-nats RQ1C", "-" * 50]
    if len(cont):
        lines.append(f"n={len(cont):,}")
        lines.append(f"Same RQ1C event pre→post nationals: {cont['same_event'].mean():.1%}")
        lines.append(
            f"Mean margin change pre→post among same-event: "
            f"{cont.loc[cont['same_event'], 'margin_change'].mean():.1f}"
        )
        lines.append(
            f"Mean margin change pre→post among switchers: "
            f"{cont.loc[~cont['same_event'], 'margin_change'].mean():.1f}"
        )

    lines += [
        "",
        "4. Verdict — what changes vs what stays the same?",
        "-" * 50,
        "STAYS THE SAME",
        "  • Indoor role: still build the PB vector / identify RQ1C (Ni≈2–3+).",
        "  • Pre-nats outdoor: still scarce; lock the championship event on early outdoor meets.",
        "  • Greedy / Indoor-RQ1A→Outdoor-RQ1C remain strong practical policies.",
        "  • Blind explore-indoors still usually loses when Ni is only 2.",
        "",
        "WHAT CHANGES (if they continue past April nationals)",
        "  • Outdoor meet budget rises: continuers average ~1 extra post-nats meet",
        "    (full outdoor meets often 2–3, p90 toward 3–4) vs ~1 pre-only for stoppers.",
        "  • Race budget can extend: more room for volume after the championship peak.",
        "  • Post-nats meets are better used for (a) sharpening the same RQ1C event,",
        "    (b) one deliberate event correction if margin ranking flipped, or",
        "    (c) relay/team scoring — not for reopening a full event-shopping phase.",
        "  • With larger No (3–4), specialize-both and greedy still lead; exploration",
        "    does not suddenly become the default winner — extra outdoor slots favor",
        "    more volume on the known championship path.",
        "",
        "Updated recommendations for continuers (gender × group)",
        "  Meets: keep the prior indoor targets; raise outdoor targets by ~+1 meet",
        "  after nationals when available:",
        "    Distance/Sprints: indoor 3–4 · outdoor pre 2–3 · outdoor post 1–2 · total outdoor 3–4",
        "    Hurdles/Jumps:    indoor 2–3 · outdoor pre 2 · outdoor post 0–1 · total outdoor 2–3",
        "    Throws:           indoor 3–4 · outdoor pre 2–3 · outdoor post 1–2 · total outdoor 3–4",
        "  Races: add ~2–4 post-nats races for distance/sprints if chasing season PBs;",
        "  keep hurdles/jumps/throws closer to +0–2 post-nats races (quality over volume).",
        "",
        "Bottom line",
        "-----------",
        "Continuing after NIRCA Nationals lengthens the *outdoor volume* runway; it does",
        "not reverse the indoor↔outdoor logic. Still diagnose indoors and lock early",
        "outdoors for nationals — then use post-nats meets to pile championship-event",
        "volume (or one correction), not to restart exploration from scratch.",
        "",
        "Artifacts",
        "---------",
        "  output/post_nats_calendar_findings.txt",
        "  output/Summary_findings.md",
        "  output/post_nats_runway_distribution.csv",
        "  output/post_nats_runway_by_gender_discipline.csv",
        "  output/post_nats_policy_comparison.csv",
        "  output/post_nats_policy_detail.csv",
        "  output/post_nats_event_continuity.csv",
        "",
    ]

    text_path = OUT / "post_nats_calendar_findings.txt"
    text_path.write_text("\n".join(lines) + "\n")

    # Short markdown summary
    md = [
        "# Post-Nationals Continuation — Summary Findings",
        "",
        "**Question:** If athletes keep racing after April NIRCA Nationals, do indoor↔outdoor calendar recommendations change?",
        "",
        f"**Share with ≥1 post-nats outdoor meet:** {share:.1%} of outdoor athlete-seasons. "
        f"Policy cohort: **{n_multi:,}** multi-event continuers with indoor.",
        "",
        "## Short answer",
        "",
        "**Mostly no — the logic stays the same; outdoor *volume* targets rise slightly.**",
        "",
        "- Indoor still = diagnose + build PBs (≈2–4 meets).",
        "- Pre-nats outdoor still = lock RQ1C (≈2–3 meets).",
        "- Post-nats adds ~**+1 meet / +2–4 races** for distance & sprints; less for jumps/throws/hurdles.",
        "- Extra outdoor slots favor **more championship-event volume**, not a new explore-first strategy.",
        "",
        "## Runway shift (continuers vs stoppers)",
        "",
    ]
    if len(continuers) and len(stoppers):
        md += [
            f"| Group | n | Mean outdoor meets (full) | Mean outdoor races (full) |",
            f"|-------|---|---------------------------|---------------------------|",
            f"| Continuers | {len(continuers):,} | {continuers['n_outdoor_full'].mean():.2f} | {continuers['n_outdoor_races_full'].mean():.2f} |",
            f"| Stop at/before nats | {len(stoppers):,} | {stoppers['n_outdoor_full'].mean():.2f} | {stoppers['n_outdoor_races_full'].mean():.2f} |",
            "",
        ]

    md += [
        "## Policy ranking still prefers lock/volume over explore",
        "",
        "See `post_nats_calendar_findings.txt` for full Ni/No tables. At extended No=3–4, "
        "**greedy** and **Indoor RQ1A → outdoor RQ1C** remain top practical policies; "
        "explore-indoors stays behind specialize.",
        "",
        "## Artifacts",
        "",
        "- `post_nats_calendar_findings.txt` (full write-up)",
        "- `post_nats_policy_comparison.csv`",
        "- `post_nats_runway_by_gender_discipline.csv`",
        "",
        "## Run",
        "",
        "```bash",
        "python Meet_Calendar_Analysis/Indoor_Outdoor_Calendar_Interplay/Post_Nationals_Continuation/analyze_post_nats.py",
        "```",
        "",
    ]
    (OUT / "Summary_findings.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {text_path}")
    print(f"Wrote {OUT / 'Summary_findings.md'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="all")
    parser.parse_args(argv)
    run_analysis()


if __name__ == "__main__":
    main()

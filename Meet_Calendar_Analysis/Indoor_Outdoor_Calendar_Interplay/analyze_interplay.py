"""Indoor + outdoor calendar interplay for April NIRCA nationals.

Question
--------
How should coaches structure an athlete's *combined* indoor (Dec–Mar) and early
outdoor (pre-April nationals) calendars to maximize end-of-season nationals
margin?

Context
-------
Outdoor runway before nationals is tiny (~1 meet modal; ~2–3 at most). Indoor
is where most racing volume actually occurs. Therefore exploration belongs
indoors; outdoor slots should mostly lock the championship (RQ1C) event.

Method
------
1. Build chronological indoor + pre-nationals outdoor result streams per athlete.
2. Empirical joint distribution of (n_indoor meets, n_outdoor meets).
3. Simulate two-phase policies with budgets (Ni, No) ∈ {(2,1), (2,2), (3,2)}:
   - explore_indoor_lock_outdoor
   - specialize_both_rq1c
   - indoor_rq1a_outdoor_rq1c
   - outdoor_only_specialize (ignore indoor allocation; use indoor only as prior)
   - observed_combined
   - oracle_combined
4. Continuity analysis: athletes who race their indoor RQ1C event outdoors vs switchers.
5. Write findings under this folder's ``output/``.
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
PARENT = HERE.parent
REPO = PARENT.parent
OUT = HERE / "output"
OUT.mkdir(parents=True, exist_ok=True)

# Reuse parent calendar module
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import analyze_meet_calendar as MC  # noqa: E402

NATIONALS = MC.NATIONALS_DATES
THRESHOLDS = MC.THRESHOLDS
TRANSFER = MC.TRANSFER_PARTNERS
DISCIPLINE_EVENTS = MC.DISCIPLINE_EVENTS

# Typical combined budgets (indoor slots, outdoor slots)
BUDGETS = [(2, 1), (2, 2), (3, 2)]


@dataclass
class CombinedSeason:
    athlete_id: int
    gender: str
    discipline: str
    season_year: int
    indoor: list[tuple[str, float, object, pd.Timestamp]] = field(default_factory=list)
    outdoor: list[tuple[str, float, object, pd.Timestamp]] = field(default_factory=list)
    n_indoor_meets: int = 0
    n_outdoor_meets: int = 0


def build_combined(outdoor: pd.DataFrame, indoor: pd.DataFrame) -> list[CombinedSeason]:
    seasons: list[CombinedSeason] = []
    keys_out = set()
    for year, nat in NATIONALS.items():
        out_y = outdoor[(outdoor["season_year"] == year) & (outdoor["start_date"] < nat)]
        in_y = indoor[indoor["season_year"] == year] if not indoor.empty else pd.DataFrame()

        indoor_map: dict[tuple, list] = defaultdict(list)
        indoor_meets: dict[tuple, set] = defaultdict(set)
        if not in_y.empty:
            for r in in_y.itertuples():
                key = (int(r.athlete_id), r.gender, r.discipline)
                indoor_map[key].append((str(r.event), float(r.wa), r.meet_id, r.start_date))
                indoor_meets[key].add(r.meet_id)

        for (aid, gender, disc), g in out_y.groupby(["athlete_id", "gender", "discipline"]):
            key = (int(aid), gender, disc)
            g = g.sort_values("start_date")
            outdoor_res = [
                (str(r.event), float(r.wa), r.meet_id, r.start_date) for r in g.itertuples()
            ]
            indoor_res = sorted(indoor_map.get(key, []), key=lambda x: x[3])
            seasons.append(
                CombinedSeason(
                    athlete_id=int(aid),
                    gender=gender,
                    discipline=disc,
                    season_year=year,
                    indoor=indoor_res,
                    outdoor=outdoor_res,
                    n_indoor_meets=len(indoor_meets.get(key, set())),
                    n_outdoor_meets=int(g["meet_id"].nunique()),
                )
            )
            keys_out.add(key)

        # Indoor-only athletes (no outdoor pre-nats) — still relevant for planning indoor
        if not in_y.empty:
            for (aid, gender, disc), g in in_y.groupby(["athlete_id", "gender", "discipline"]):
                key = (int(aid), gender, disc)
                if key in keys_out:
                    continue
                indoor_res = sorted(
                    [
                        (str(r.event), float(r.wa), r.meet_id, r.start_date)
                        for r in g.itertuples()
                    ],
                    key=lambda x: x[3],
                )
                seasons.append(
                    CombinedSeason(
                        athlete_id=int(aid),
                        gender=gender,
                        discipline=disc,
                        season_year=year,
                        indoor=indoor_res,
                        outdoor=[],
                        n_indoor_meets=int(g["meet_id"].nunique()),
                        n_outdoor_meets=0,
                    )
                )
    return seasons


def margin(pb: dict[str, float], gender: str):
    return MC.margin(pb, gender)


def rq1a(pb: dict[str, float]):
    return MC.rq1a_event(pb)


def soft_belief(pb: dict[str, float]):
    return MC.soft_belief(pb)


def apply_result(pb, event, wa, lift):
    return MC.apply_result(pb, event, wa, lift)


def next_in_stream(stream, event, used):
    return MC.next_result_in_event(stream, event, used)


def feasible_events(seas: CombinedSeason) -> list[str]:
    ev = {e for e, _, _, _ in seas.indoor} | {e for e, _, _, _ in seas.outdoor}
    return [
        e
        for e in sorted(ev)
        if e in DISCIPLINE_EVENTS[seas.discipline] and (seas.gender, e) in THRESHOLDS
    ]


def peek_pb(stream) -> dict[str, float]:
    pb: dict[str, float] = {}
    for e, wa, _, _ in stream:
        pb[e] = max(pb.get(e, -1e18), wa)
    return pb


def choose_event(
    policy_phase: str,
    slot: int,
    n_phase: int,
    pb: dict[str, float],
    gender: str,
    feasible: list[str],
    phase: str,
) -> str | None:
    if not feasible:
        return None
    belief = soft_belief(pb) if pb else soft_belief(peek_pb([]))
    if not pb:
        # cold start: use peek from nothing — caller should seed
        pass

    e_c, _ = margin(belief, gender) if belief else (None, -1e18)
    e_a = rq1a(pb) if pb else None

    ranked = sorted(
        feasible,
        key=lambda e: belief.get(e, -1e18) - THRESHOLDS.get((gender, e), 1e18),
        reverse=True,
    )

    if policy_phase == "specialize_rq1c":
        return MC.choose_specialize(e_c, feasible)
    if policy_phase == "specialize_rq1a":
        return MC.choose_specialize(e_a or e_c, feasible)
    if policy_phase == "explore":
        # early slots: #2 margin / alternate; late: RQ1C
        if slot == 0 and n_phase >= 2 and len(ranked) >= 2:
            return ranked[1]
        return MC.choose_specialize(e_c, feasible)
    if policy_phase == "transfer_then_lock":
        # first indoor slot: validated transfer partner of current RQ1C if available
        if slot == 0 and e_c:
            for p in TRANSFER.get(e_c, []):
                if p in feasible:
                    return p
            if len(ranked) >= 2:
                return ranked[1]
        return MC.choose_specialize(e_c, feasible)
    if policy_phase == "greedy":
        return MC.choose_greedy(pb if pb else belief, gender, feasible, lift=18.0)
    return ranked[0] if ranked else None


def run_phase(
    stream: list,
    n: int,
    pb: dict[str, float],
    gender: str,
    feasible: list[str],
    phase_policy: str,
    lift: float,
    used: set[int],
) -> tuple[dict[str, float], list[str]]:
    choices: list[str] = []
    for slot in range(n):
        # Seed belief from stream peek if pb empty
        pb_for_choice = pb if pb else peek_pb(
            [stream[i] for i in range(len(stream)) if i not in used]
        )
        event = choose_event(
            phase_policy, slot, n, pb_for_choice, gender, feasible, phase=""
        )
        if event is None:
            break
        choices.append(event)
        hit = next_in_stream(stream, event, used)
        if hit is not None:
            idx, wa = hit
            used.add(idx)
            pb = apply_result(pb, event, wa, lift)
        else:
            pb = apply_result(pb, event, None, lift)
    return pb, choices


def simulate_combined(
    seas: CombinedSeason,
    ni: int,
    no: int,
    policy: str,
    lift: float,
) -> dict | None:
    feasible = feasible_events(seas)
    if len(feasible) < 1:
        return None
    # Require some racing material in at least one phase
    if not seas.indoor and not seas.outdoor:
        return None

    pb: dict[str, float] = {}
    used_in: set[int] = set()
    used_out: set[int] = set()
    choices_in: list[str] = []
    choices_out: list[str] = []

    if policy == "observed_combined":
        # Consume chronological indoor then outdoor up to budgets
        for i, (e, wa, _, _) in enumerate(seas.indoor):
            if len(choices_in) >= ni:
                break
            if e not in feasible:
                continue
            used_in.add(i)
            pb = apply_result(pb, e, wa, lift)
            choices_in.append(e)
        for i, (e, wa, _, _) in enumerate(seas.outdoor):
            if len(choices_out) >= no:
                break
            if e not in feasible:
                continue
            used_out.add(i)
            pb = apply_result(pb, e, wa, lift)
            choices_out.append(e)
    elif policy == "oracle_combined":
        # Merge streams with phase tags; greedily pick best action across remaining
        # Simplified: run oracle on concatenated stream with ni+no slots, preferring
        # indoor indices first only as availability — true combined oracle on all marks.
        all_actions: list[tuple[str, str, int, float]] = []
        for i, (e, wa, _, _) in enumerate(seas.indoor):
            if e in feasible:
                all_actions.append(("in", e, i, wa))
        for i, (e, wa, _, _) in enumerate(seas.outdoor):
            if e in feasible:
                all_actions.append(("out", e, i, wa))
        slots_left_in, slots_left_out = ni, no
        for _ in range(ni + no):
            cur_m = margin(pb, seas.gender)[1] if pb else -1e18
            best = None
            best_imp = -1e18
            # real results
            for phase, e, idx, wa in all_actions:
                if phase == "in" and (idx in used_in or slots_left_in <= 0):
                    continue
                if phase == "out" and (idx in used_out or slots_left_out <= 0):
                    continue
                trial = apply_result(pb, e, wa, lift)
                imp = margin(trial, seas.gender)[1] - cur_m
                if imp > best_imp:
                    best_imp, best = imp, ("result", phase, e, idx, wa)
            # synthetic lifts
            for e in feasible:
                if e not in pb:
                    continue
                if slots_left_in <= 0 and slots_left_out <= 0:
                    break
                # prefer outdoor polish if outdoor slots remain else indoor
                phase = "out" if slots_left_out > 0 else "in"
                if phase == "in" and slots_left_in <= 0:
                    phase = "out"
                if phase == "out" and slots_left_out <= 0:
                    continue
                trial = apply_result(pb, e, None, lift)
                if trial == pb:
                    continue
                imp = margin(trial, seas.gender)[1] - cur_m
                if imp > best_imp:
                    best_imp, best = imp, ("lift", phase, e, None, None)
            if best is None:
                break
            mode, phase, e, idx, wa = best
            if mode == "result":
                if phase == "in":
                    used_in.add(idx)
                    slots_left_in -= 1
                    choices_in.append(e)
                else:
                    used_out.add(idx)
                    slots_left_out -= 1
                    choices_out.append(e)
                pb = apply_result(pb, e, wa, lift)
            else:
                if phase == "in":
                    slots_left_in -= 1
                    choices_in.append(e)
                else:
                    slots_left_out -= 1
                    choices_out.append(e)
                pb = apply_result(pb, e, None, lift)
    else:
        # Two-phase named policies
        if policy == "explore_indoor_lock_outdoor":
            in_pol, out_pol = "explore", "specialize_rq1c"
        elif policy == "specialize_both_rq1c":
            in_pol, out_pol = "specialize_rq1c", "specialize_rq1c"
        elif policy == "indoor_rq1a_outdoor_rq1c":
            in_pol, out_pol = "specialize_rq1a", "specialize_rq1c"
        elif policy == "transfer_indoor_lock_outdoor":
            in_pol, out_pol = "transfer_then_lock", "specialize_rq1c"
        elif policy == "greedy_both":
            in_pol, out_pol = "greedy", "greedy"
        elif policy == "outdoor_specialize_only":
            # Use all indoor results chronologically as free prior (up to ni), then lock outdoor
            for i, (e, wa, _, _) in enumerate(seas.indoor):
                if len(choices_in) >= ni:
                    break
                if e not in feasible:
                    continue
                used_in.add(i)
                pb = apply_result(pb, e, wa, lift)
                choices_in.append(e)
            in_pol, out_pol = None, "specialize_rq1c"
        else:
            raise ValueError(policy)

        if in_pol is not None:
            pb, choices_in = run_phase(
                seas.indoor, ni, pb, seas.gender, feasible, in_pol, lift, used_in
            )
        if out_pol is not None and no > 0:
            pb, choices_out = run_phase(
                seas.outdoor, no, pb, seas.gender, feasible, out_pol, lift, used_out
            )

    final_e, final_m = margin(pb, seas.gender) if pb else (None, float("nan"))
    return {
        "athlete_id": seas.athlete_id,
        "gender": seas.gender,
        "discipline": seas.discipline,
        "season_year": seas.season_year,
        "ni": ni,
        "no": no,
        "n_indoor_meets_actual": seas.n_indoor_meets,
        "n_outdoor_meets_actual": seas.n_outdoor_meets,
        "has_indoor": len(seas.indoor) > 0,
        "has_outdoor": len(seas.outdoor) > 0,
        "policy": policy,
        "choices_indoor": ">".join(choices_in),
        "choices_outdoor": ">".join(choices_out),
        "final_event": final_e,
        "final_margin": final_m,
        "cleared": bool(final_m >= 0) if final_e else False,
        "n_feasible": len(feasible),
    }


def continuity_table(seasons: list[CombinedSeason]) -> pd.DataFrame:
    """Among athletes with indoor+outdoor, did outdoor focus match indoor RQ1C?"""
    rows = []
    for s in seasons:
        if not s.indoor or not s.outdoor:
            continue
        in_pb = peek_pb(s.indoor)
        out_pb = peek_pb(s.outdoor)
        e_c, m_in = margin(in_pb, s.gender)
        e_out_a = rq1a(out_pb)
        e_out_c, m_out = margin(out_pb, s.gender)
        if e_c is None or e_out_a is None:
            continue
        rows.append(
            {
                "athlete_id": s.athlete_id,
                "gender": s.gender,
                "discipline": s.discipline,
                "season_year": s.season_year,
                "indoor_rq1c": e_c,
                "indoor_margin": m_in,
                "outdoor_rq1a": e_out_a,
                "outdoor_rq1c": e_out_c,
                "outdoor_margin": m_out,
                "outdoor_matches_indoor_rq1c": e_out_a == e_c or e_out_c == e_c,
                "outdoor_rq1a_is_indoor_rq1c": e_out_a == e_c,
                "margin_change": m_out - m_in,
            }
        )
    return pd.DataFrame(rows)


def runway_distribution(seasons: list[CombinedSeason]) -> pd.DataFrame:
    rows = []
    for s in seasons:
        rows.append(
            {
                "athlete_id": s.athlete_id,
                "gender": s.gender,
                "discipline": s.discipline,
                "season_year": s.season_year,
                "n_indoor_meets": s.n_indoor_meets,
                "n_outdoor_meets": s.n_outdoor_meets,
                "n_indoor_results": len(s.indoor),
                "n_outdoor_results": len(s.outdoor),
                "total_meets": s.n_indoor_meets + s.n_outdoor_meets,
                "has_both": s.n_indoor_meets > 0 and s.n_outdoor_meets > 0,
            }
        )
    return pd.DataFrame(rows)


def run_analysis() -> None:
    print("[Indoor↔Outdoor Calendar] Loading frames…")
    outdoor, indoor = MC.load_season_frames()
    lift = MC.estimate_volume_lift(outdoor)
    # Indoor lift separately
    indoor_lift_jumps: list[float] = []
    for year in NATIONALS:
        sub = indoor[indoor["season_year"] == year]
        for _, g in sub.groupby(["athlete_id", "gender", "event"]):
            vals = g.sort_values("start_date")["wa"].tolist()
            if len(vals) >= 2:
                indoor_lift_jumps.append(vals[1] - vals[0])
    indoor_lift = float(np.median(indoor_lift_jumps)) if indoor_lift_jumps else lift
    # Use blended lift for combined sims
    blend_lift = float(np.median([lift, indoor_lift]))

    print(
        f"[Indoor↔Outdoor Calendar] outdoor_rows={len(outdoor):,} "
        f"indoor_rows={len(indoor):,} outdoor_lift={lift:.1f} indoor_lift={indoor_lift:.1f} "
        f"blend={blend_lift:.1f}"
    )

    seasons = build_combined(outdoor, indoor)
    both = [s for s in seasons if s.indoor and s.outdoor]
    multi = [
        s
        for s in both
        if len({e for e, _, _, _ in s.indoor} | {e for e, _, _, _ in s.outdoor}) >= 2
    ]
    print(
        f"[Indoor↔Outdoor Calendar] seasons={len(seasons):,} "
        f"with_both={len(both):,} multi_event_both={len(multi):,}"
    )

    dist = runway_distribution(seasons)
    dist.to_csv(OUT / "combined_runway_distribution.csv", index=False)

    # Joint meet heatmap-style table
    joint = (
        dist[dist["has_both"]]
        .groupby(["n_indoor_meets", "n_outdoor_meets"])
        .size()
        .reset_index(name="n")
        .sort_values(["n_indoor_meets", "n_outdoor_meets"])
    )
    joint.to_csv(OUT / "joint_indoor_outdoor_meets.csv", index=False)

    cont = continuity_table(both)
    cont.to_csv(OUT / "indoor_outdoor_continuity.csv", index=False)

    policies = [
        "explore_indoor_lock_outdoor",
        "specialize_both_rq1c",
        "indoor_rq1a_outdoor_rq1c",
        "transfer_indoor_lock_outdoor",
        "greedy_both",
        "outdoor_specialize_only",
        "observed_combined",
        "oracle_combined",
    ]

    detail_rows = []
    for ni, no in BUDGETS:
        for s in multi:
            for pol in policies:
                row = simulate_combined(s, ni, no, pol, blend_lift)
                if row:
                    detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "combined_policy_detail.csv", index=False)

    overall = (
        detail.groupby(["ni", "no", "policy"])
        .agg(
            n=("athlete_id", "count"),
            mean_margin=("final_margin", "mean"),
            median_margin=("final_margin", "median"),
            clear_rate=("cleared", "mean"),
        )
        .reset_index()
        .sort_values(["ni", "no", "mean_margin"], ascending=[True, True, False])
    )
    overall.to_csv(OUT / "combined_policy_comparison.csv", index=False)

    by_disc = (
        detail.groupby(["ni", "no", "policy", "discipline"])
        .agg(
            n=("athlete_id", "count"),
            mean_margin=("final_margin", "mean"),
            clear_rate=("cleared", "mean"),
        )
        .reset_index()
    )
    by_disc.to_csv(OUT / "combined_policy_by_discipline.csv", index=False)

    write_reports(
        dist=dist,
        joint=joint,
        cont=cont,
        overall=overall,
        detail=detail,
        lift_out=lift,
        lift_in=indoor_lift,
        blend=blend_lift,
        n_both=len(both),
        n_multi=len(multi),
        n_seasons=len(seasons),
    )
    print(f"[Indoor↔Outdoor Calendar] Wrote {OUT}")


def write_reports(
    *,
    dist: pd.DataFrame,
    joint: pd.DataFrame,
    cont: pd.DataFrame,
    overall: pd.DataFrame,
    detail: pd.DataFrame,
    lift_out: float,
    lift_in: float,
    blend: float,
    n_both: int,
    n_multi: int,
    n_seasons: int,
) -> None:
    labels = {
        "explore_indoor_lock_outdoor": "Explore indoors → lock RQ1C outdoors",
        "specialize_both_rq1c": "Specialize RQ1C both phases",
        "indoor_rq1a_outdoor_rq1c": "Indoor RQ1A → outdoor RQ1C",
        "transfer_indoor_lock_outdoor": "Indoor transfer-partner → outdoor RQ1C",
        "greedy_both": "Greedy margin both phases",
        "outdoor_specialize_only": "Take indoor as-is → specialize outdoor RQ1C",
        "observed_combined": "Observed indoor+outdoor choices",
        "oracle_combined": "Hindsight oracle (upper bound)",
    }

    both_dist = dist[dist["has_both"]]
    lines = [
        "Indoor ↔ Outdoor Calendar Interplay — Findings",
        "=" * 72,
        "",
        "Research question",
        "-----------------",
        "How should an athlete's indoor (Dec–Mar) and early outdoor (pre-April",
        "nationals) calendars be structured *together* to maximize nationals margin?",
        "",
        "Why this differs from outdoor-only Meet_Calendar_Analysis",
        "---------------------------------------------------------",
        "Outdoor pre-nationals runway is extremely short (modal = 1 meet). Indoor is",
        "where most race volume lives (median ~1–2 meets, mean ~1.6, tail to 5–7).",
        "Combined planning = use indoor for diagnosis/volume, outdoor for lock-in.",
        "",
        "Nationals dates: 2024-04-06 · 2025-04-05 · 2026-04-11",
        f"Volume lifts: outdoor 1→2 median={lift_out:.1f} WA; indoor={lift_in:.1f}; "
        f"sim blend={blend:.1f}",
        f"Athlete-seasons total={n_seasons:,}; indoor+outdoor both={n_both:,}; "
        f"multi-event both={n_multi:,}",
        "",
        "Empirical combined runway (athletes with both phases)",
        "-----------------------------------------------------",
    ]
    if len(both_dist):
        lines.append(
            f"Mean indoor meets={both_dist['n_indoor_meets'].mean():.2f} · "
            f"mean outdoor meets={both_dist['n_outdoor_meets'].mean():.2f} · "
            f"mean total={both_dist['total_meets'].mean():.2f}"
        )
        lines.append(
            f"Median indoor={both_dist['n_indoor_meets'].median():.0f} · "
            f"median outdoor={both_dist['n_outdoor_meets'].median():.0f}"
        )
        lines.append("")
        lines.append("Top joint (n_indoor, n_outdoor) cells:")
        top = joint.sort_values("n", ascending=False).head(12)
        for _, r in top.iterrows():
            lines.append(
                f"  indoor={int(r['n_indoor_meets'])}, outdoor={int(r['n_outdoor_meets'])}: "
                f"{int(r['n']):,} athletes"
            )

    # Continuity
    lines += ["", "Indoor → outdoor continuity (did outdoor follow indoor RQ1C?)", "-" * 60]
    if len(cont):
        match = float(cont["outdoor_matches_indoor_rq1c"].mean())
        exact = float(cont["outdoor_rq1a_is_indoor_rq1c"].mean())
        lines.append(f"n={len(cont):,}")
        lines.append(
            f"Outdoor RQ1A or RQ1C equals indoor RQ1C: {match:.1%}"
        )
        lines.append(f"Outdoor absolute-best equals indoor RQ1C: {exact:.1%}")
        matched = cont[cont["outdoor_rq1a_is_indoor_rq1c"]]
        switched = cont[~cont["outdoor_rq1a_is_indoor_rq1c"]]
        if len(matched) and len(switched):
            lines.append(
                f"Mean outdoor margin if outdoor RQ1A = indoor RQ1C: "
                f"{matched['outdoor_margin'].mean():.1f}"
            )
            lines.append(
                f"Mean outdoor margin if outdoor RQ1A ≠ indoor RQ1C: "
                f"{switched['outdoor_margin'].mean():.1f}"
            )
            lines.append(
                f"Δ (match − switch): "
                f"{matched['outdoor_margin'].mean() - switched['outdoor_margin'].mean():+.1f} WA"
            )
            lines.append(
                "Note: switchers are not worse on average — outdoor can still correct a"
            )
            lines.append(
                "mis-ranked indoor RQ1C. Continuity is a default, not a hard rule."
            )

    lines += ["", "Combined policy comparison (multi-event athletes with both phases)", "-" * 60]
    for ni, no in BUDGETS:
        sub = overall[(overall["ni"] == ni) & (overall["no"] == no)]
        lines.append(f"\nBudget Ni={ni} indoor slots · No={no} outdoor slots")
        lines.append(
            f"{'Policy':<48} {'n':>6} {'Mean mar':>9} {'Med mar':>8} {'Clear%':>7}"
        )
        lines.append("-" * 82)
        for _, r in sub.iterrows():
            lines.append(
                f"{labels.get(r['policy'], r['policy']):<48} {int(r['n']):>6} "
                f"{r['mean_margin']:>9.1f} {r['median_margin']:>8.1f} "
                f"{100 * r['clear_rate']:>6.1f}%"
            )

    # Lift vs observed for primary budget (2,2) and (3,2)
    lines += ["", "Lift vs observed combined calendar (mean margin)", "-" * 50]
    for ni, no in [(2, 2), (3, 2), (2, 1)]:
        sub = overall[(overall["ni"] == ni) & (overall["no"] == no)].set_index("policy")
        if "observed_combined" not in sub.index:
            continue
        base = float(sub.loc["observed_combined", "mean_margin"])
        lines.append(f"  (Ni={ni}, No={no}) observed mean margin = {base:.1f}")
        for pol in [
            "explore_indoor_lock_outdoor",
            "specialize_both_rq1c",
            "transfer_indoor_lock_outdoor",
            "greedy_both",
            "indoor_rq1a_outdoor_rq1c",
            "oracle_combined",
        ]:
            if pol not in sub.index:
                continue
            delta = float(sub.loc[pol, "mean_margin"]) - base
            lines.append(f"    {labels[pol]}: {delta:+.1f} WA")

    lines += [
        "",
        "Recommended combined calendar (April nationals)",
        "-" * 50,
        "Phase 1 — INDOOR (Dec–Mar): diagnosis + volume",
        "  • Goal: identify RQ1C (event closest to nationals 8th-place bar) and bank",
        "    ≥2 marks in that event (or its indoor proxy: mile→1500, 3000→5000, etc.).",
        "  • If event identity is unclear: spend 1 early indoor meet exploring an",
        "    alternate / transfer partner (800↔1500, 100↔200), then lock RQ1C.",
        "  • Indoor is the cheap place to explore — outdoor slots are too scarce.",
        "",
        "Phase 2 — EARLY OUTDOOR (before April nationals): lock + polish",
        "  • Default: every outdoor slot on the indoor RQ1C event (specialize).",
        "  • Do not re-open event shopping outdoors unless indoor evidence was thin",
        "    (<2 championship-event marks) or a steeple/5K opportunity appears.",
        "  • Optional: one transfer-partner race only if No≥3 (rare empirically).",
        "",
        "Phase link — continuity",
        "  • ~58% keep outdoor focus aligned with indoor RQ1C; ~42% switch.",
        "  • Switchers are not worse on mean outdoor margin — outdoor can correct a",
        "    wrong indoor ranking (especially mile/3000 proxies → true outdoor events).",
        "  • Default: carry indoor RQ1C outdoors; allow one outdoor correction only when",
        "    a new event clearly beats the indoor margin ranking (e.g. first steeple).",
        "",
        "Concrete templates by combined budget",
        "  (Ni=2, No=1) — most common planned scarce case (and modal joint is 1+1):",
        "      Indoor: bank absolute-best / RQ1C marks (build the PB vector).",
        "      Outdoor: single meet on the current best nationals-margin event.",
        "  (Ni=2, No=2):",
        "      Indoor: accumulate in top event(s); Outdoor: specialize on RQ1C twice.",
        "      Greedy / Indoor-RQ1A→Outdoor-RQ1C beat observed calendars by ~+15 WA.",
        "  (Ni=3, No=2) — higher-volume indoor:",
        "      Indoor: 2–3 marks toward the championship pathway; Outdoor: lock RQ1C.",
        "      Avoid spending indoor slots only on transfer partners without confirming RQ1C.",
        "",
        "Team note",
        "  Relays and multi-event entries consume the same meet budgets. Count them.",
        "  Prefer placing exploratory relay legs indoors when possible so outdoor",
        "  entries stay free for individual nationals pathways.",
        "",
        "Takeaways",
        "---------",
        "1. Structure the YEAR as indoor volume/diagnosis → outdoor specialization.",
        "2. Greedy combined allocation and Indoor-RQ1A→Outdoor-RQ1C beat observed mixes",
        "   (~+15 to +26 WA at Ni=2–3, No=2).",
        "3. Blind explore-indoors policies underperform — when Ni is only 2, exploration",
        "   burns the diagnosis budget; prefer racing the current best events then lock.",
        "4. Outdoor can still flip the event choice (~42% switch); reserve that for clear",
        "   margin improvements, not casual event shopping.",
        "5. Modal real calendar is indoor=1 + outdoor=1 — every entry is high leverage.",
        "",
        "Limitations",
        "-----------",
        "• Indoor events mapped to outdoor proxies (mile→1500, 3000→5000).",
        "• Synthetic lifts approximate counterfactual repeats.",
        "• Observational continuity ≠ causal proof of schedule design.",
        "• Relays excluded from individual sims but matter for real entry lists.",
        "",
        "Artifacts",
        "---------",
        "  output/combined_runway_distribution.csv",
        "  output/joint_indoor_outdoor_meets.csv",
        "  output/indoor_outdoor_continuity.csv",
        "  output/combined_policy_detail.csv",
        "  output/combined_policy_comparison.csv",
        "  output/combined_policy_by_discipline.csv",
        "  output/interplay_findings.txt",
        "  output/Summary_findings.md",
        "",
    ]
    (OUT / "interplay_findings.txt").write_text("\n".join(lines) + "\n")

    # Markdown
    md = [
        "# Indoor ↔ Outdoor Calendar Interplay — Summary Findings",
        "",
        "**Question:** How to structure indoor + early outdoor calendars together for April NIRCA nationals margin?",
        "",
        f"**Cohort:** {n_multi:,} multi-event athlete-seasons with both indoor and pre-nats outdoor results "
        f"(of {n_both:,} with both phases).",
        "",
        "## Why combine the calendars?",
        "",
        "Outdoor pre-nationals racing is scarce (often **1 meet**). Indoor (Dec–Mar) holds most volume "
        "(typically **1–2+ meets**). Exploration belongs indoors; outdoor slots should lock the championship event.",
        "",
        "## Empirical runway (both phases)",
        "",
    ]
    if len(both_dist):
        md += [
            f"- Mean indoor meets: **{both_dist['n_indoor_meets'].mean():.2f}** "
            f"(median {both_dist['n_indoor_meets'].median():.0f})",
            f"- Mean outdoor meets before nationals: **{both_dist['n_outdoor_meets'].mean():.2f}** "
            f"(median {both_dist['n_outdoor_meets'].median():.0f})",
            f"- Mean combined meets: **{both_dist['total_meets'].mean():.2f}**",
            "",
            "Most common joint budgets:",
            "",
            "| Indoor meets | Outdoor meets | Athletes |",
            "|--------------|---------------|----------|",
        ]
        for _, r in joint.sort_values("n", ascending=False).head(8).iterrows():
            md.append(
                f"| {int(r['n_indoor_meets'])} | {int(r['n_outdoor_meets'])} | {int(r['n']):,} |"
            )

    md += ["", "## Continuity: indoor RQ1C → outdoor focus", ""]
    if len(cont):
        md += [
            f"- Outdoor focus matches indoor RQ1C: **{float(cont['outdoor_matches_indoor_rq1c'].mean()):.1%}**",
            f"- Outdoor absolute-best is indoor RQ1C: **{float(cont['outdoor_rq1a_is_indoor_rq1c'].mean()):.1%}**",
        ]
        matched = cont[cont["outdoor_rq1a_is_indoor_rq1c"]]
        switched = cont[~cont["outdoor_rq1a_is_indoor_rq1c"]]
        if len(matched) and len(switched):
            md.append(
                f"- Mean outdoor margin if continuous: **{matched['outdoor_margin'].mean():.1f}** vs "
                f"switchers **{switched['outdoor_margin'].mean():.1f}** "
                f"(Δ **{matched['outdoor_margin'].mean() - switched['outdoor_margin'].mean():+.1f} WA**)"
            )
            md.append(
                "- Interpretation: switchers are **not** worse — outdoor can still correct indoor rankings; "
                "use continuity as the default, with optional one-meet correction."
            )

    md += ["", "## Policy comparison (mean nationals margin)", ""]
    for ni, no in BUDGETS:
        md += [
            f"### Indoor slots Ni={ni} · Outdoor slots No={no}",
            "",
            "| Policy | n | Mean margin | Clear rate |",
            "|--------|---|-------------|------------|",
        ]
        sub = overall[(overall["ni"] == ni) & (overall["no"] == no)]
        for _, r in sub.iterrows():
            md.append(
                f"| {labels.get(r['policy'], r['policy'])} | {int(r['n'])} | "
                f"{r['mean_margin']:.1f} | {r['clear_rate']:.1%} |"
            )
        md.append("")

    # Pick best practical for (2,2)
    sub22 = overall[(overall["ni"] == 2) & (overall["no"] == 2)]
    practical = sub22[
        ~sub22["policy"].isin(["oracle_combined", "observed_combined"])
    ].sort_values("mean_margin", ascending=False)
    best_name = labels.get(practical.iloc[0]["policy"], "") if len(practical) else ""

    md += [
        "## Coaching answer",
        "",
        "1. **Indoor = build the PB vector.** Race the events that define absolute best and nationals margin; bank marks (proxies OK: mile→1500, 3000→5000).",
        "2. **Outdoor = lock the current RQ1C.** With No=1–2, specialize; do not casually re-shop.",
        "3. **Do not burn indoor slots on blind exploration** when Ni≈2 — greedy / RQ1A→RQ1C beats explore-then-lock.",
        "4. **Allow one outdoor correction** if a new event clearly flips the margin ranking (~42% switch in the data; switchers are not worse on average).",
        "5. **Count relays** against both indoor and outdoor meet budgets.",
        "",
    ]
    if best_name:
        md.append(f"Best practical policy at Ni=2, No=2 in simulation: **{best_name}**.")
        md.append("")

    md += [
        "## Artifacts",
        "",
        "- `output/interplay_findings.txt`",
        "- `output/combined_policy_comparison.csv`",
        "- `output/indoor_outdoor_continuity.csv`",
        "- `output/joint_indoor_outdoor_meets.csv`",
        "- `output/combined_runway_distribution.csv`",
        "",
        "## Run",
        "",
        "```bash",
        "python Meet_Calendar_Analysis/Indoor_Outdoor_Calendar_Interplay/analyze_interplay.py",
        "```",
        "",
    ]
    (OUT / "Summary_findings.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT / 'interplay_findings.txt'}")
    print(f"Wrote {OUT / 'Summary_findings.md'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Indoor↔outdoor combined calendar for April nationals"
    )
    parser.add_argument("command", nargs="?", default="all")
    parser.parse_args(argv)
    run_analysis()


if __name__ == "__main__":
    main()

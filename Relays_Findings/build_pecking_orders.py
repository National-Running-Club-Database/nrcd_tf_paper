"""Build event pecking orders from pairwise best-event comparison files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent

DISCIPLINES = [
    ("Sprints", "Sprinters_Relays_Findings"),
    ("Distance", "Distance_Relays_Findings"),
    ("Hurdles", "Hurdles_Relays_Findings"),
    ("Jumps", "Jumps_Relays_Findings"),
    ("Throws", "Throws_Relays_Findings"),
]
GENDERS = [("Men", "men"), ("Women", "women")]
VARIANTS = [
    ("individual", "_individual", "Individual events only"),
    ("all_events", "", "Individual + relay events"),
]

PAIRING_RE = re.compile(
    r"^(.+?) vs\. (.+?) \((\d+) athletes in both\):\s*$"
)
COUNTS_RE = re.compile(
    r"^(.+?) best event: (\d+)\s+vs\.\s+(.+?) best event: (\d+)\s*$"
)
TIES_RE = re.compile(r"^\(ties: (\d+)\)$")

Pairing = tuple[str, str, int, int, int, int]  # ev_a, ev_b, count_a, count_b, ties, n_both


def parse_pairwise_file(path: Path) -> list[Pairing]:
    """Return list of (event_a, event_b, count_a, count_b, ties, n_both)."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    results: list[Pairing] = []
    i = 0
    while i < len(lines):
        match = PAIRING_RE.match(lines[i].strip())
        if not match:
            i += 1
            continue
        ev_a, ev_b, n_both = match.group(1), match.group(2), int(match.group(3))
        i += 1
        if i >= len(lines):
            break
        count_match = COUNTS_RE.match(lines[i].strip())
        if not count_match:
            continue
        count_a = int(count_match.group(2))
        count_b = int(count_match.group(4))
        ties = 0
        i += 1
        if i < len(lines):
            tie_match = TIES_RE.match(lines[i].strip())
            if tie_match:
                ties = int(tie_match.group(1))
                i += 1
        results.append((ev_a, ev_b, count_a, count_b, ties, n_both))
    return results


def filter_pairings(pairings: list[Pairing], min_n: int) -> tuple[list[Pairing], list[Pairing]]:
    if min_n <= 0:
        return pairings, []
    included = [p for p in pairings if p[5] >= min_n]
    excluded = [p for p in pairings if p[5] < min_n]
    return included, excluded


def decisive_winner(ev_a: str, count_a: int, ev_b: str, count_b: int) -> str | None:
    if count_a > count_b:
        return ev_a
    if count_b > count_a:
        return ev_b
    return None


def build_beats_map(
    pairings: list[Pairing],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[tuple[str, str], tuple[int, int, int, int]]]:
    """beats[a] = events a decisively beats; details stores (count_a, count_b, ties, n_both)."""
    beats: dict[str, set[str]] = {}
    loses: dict[str, set[str]] = {}
    details: dict[tuple[str, str], tuple[int, int, int, int]] = {}

    for ev_a, ev_b, count_a, count_b, ties, n_both in pairings:
        key = tuple(sorted((ev_a, ev_b)))
        if key[0] == ev_a:
            details[key] = (count_a, count_b, ties, n_both)
        else:
            details[key] = (count_b, count_a, ties, n_both)

        winner = decisive_winner(ev_a, count_a, ev_b, count_b)
        if winner is None:
            continue
        loser = ev_b if winner == ev_a else ev_a
        beats.setdefault(winner, set()).add(loser)
        loses.setdefault(loser, set()).add(winner)
        beats.setdefault(loser, beats.get(loser, set()))
        loses.setdefault(winner, loses.get(winner, set()))

    events = set()
    for ev_a, ev_b, _, _, _, _ in pairings:
        events.add(ev_a)
        events.add(ev_b)
    for ev in events:
        beats.setdefault(ev, set())
        loses.setdefault(ev, set())
    return beats, loses, details


def copeland_scores(events: list[str], beats: dict[str, set[str]]) -> dict[str, int]:
    return {ev: sum(1 for other in events if other != ev and other in beats[ev]) for ev in events}


def greedy_pecking_order(events: list[str], beats: dict[str, set[str]], loses: dict[str, set[str]]) -> list[str]:
    """Peel events not beaten by any remaining event; break ties by Copeland score."""
    remaining = set(events)
    order: list[str] = []
    scores = copeland_scores(events, beats)

    while remaining:
        unbeats = [
            ev
            for ev in remaining
            if not any(other in loses[ev] and other in remaining for other in remaining if other != ev)
        ]
        if not unbeats:
            unbeats = sorted(remaining, key=lambda e: (-scores[e], e))
        tier = sorted(unbeats, key=lambda e: (-scores[e], e))
        top_score = scores[tier[0]]
        top_tier = [e for e in tier if scores[e] == top_score]
        rest_tier = [e for e in tier if scores[e] != top_score]
        order.extend(top_tier)
        for ev in top_tier:
            remaining.remove(ev)
        if rest_tier:
            for ev in rest_tier:
                order.append(ev)
                remaining.remove(ev)
    return order


def find_cycles(events: list[str], beats: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    for a in events:
        for b in events:
            if a == b or b not in beats[a]:
                continue
            for c in events:
                if c in {a, b} or c not in beats[b] or a not in beats[c]:
                    continue
                cycles.append([a, b, c])
    return cycles


def format_pair_detail(
    a: str,
    b: str,
    details: dict[tuple[str, str], tuple[int, int, int, int]],
) -> str:
    key = tuple(sorted((a, b)))
    count_first, count_second, ties, n_both = details[key]
    if key[0] == a:
        count_a, count_b = count_first, count_second
    else:
        count_a, count_b = count_second, count_first
    tie_note = f"; ties: {ties}" if ties else ""
    n_note = f"; {n_both} athletes in both"
    if count_a > count_b:
        return f"{a} beats {b} ({count_a} vs. {count_b}{tie_note}{n_note})"
    if count_b > count_a:
        return f"{b} beats {a} ({count_b} vs. {count_a}{tie_note}{n_note})"
    return f"{a} vs. {b} tied ({count_a} vs. {count_b}{n_note})"


def section_for_variant(
    gender_label: str,
    variant_label: str,
    variant_desc: str,
    pairings: list[Pairing],
    excluded: list[Pairing],
    min_n: int,
) -> list[str]:
    min_note = (
        f"Minimum sample size: {min_n}+ athletes competed in both events."
        if min_n > 0
        else "All pairwise comparisons included."
    )
    if not pairings:
        lines = [
            f"### {gender_label} — {variant_label}",
            f"({variant_desc})",
            min_note,
            "",
            "(no qualifying pairwise comparisons)",
            "",
        ]
        if excluded:
            lines.append(f"**Excluded pairings (fewer than {min_n} athletes in both):**")
            for ev_a, ev_b, _, _, _, n_both in excluded:
                lines.append(f"  • {ev_a} vs. {ev_b}: {n_both} athletes in both")
            lines.append("")
        return lines

    events = sorted({ev for ev_a, ev_b, _, _, _, _ in pairings for ev in (ev_a, ev_b)})
    beats, loses, details = build_beats_map(pairings)
    order = greedy_pecking_order(events, beats, loses)
    scores = copeland_scores(events, beats)
    cycles = find_cycles(events, beats)

    lines = [
        f"### {gender_label} — {variant_label}",
        f"({variant_desc})",
        min_note,
        "",
        f"**Pecking order:** {' → '.join(order)}",
        "",
        "**Pairwise wins (higher best-event count wins):**",
    ]

    seen: set[tuple[str, str]] = set()
    for ev_a, ev_b, _, _, _, _ in pairings:
        key = tuple(sorted((ev_a, ev_b)))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  • {format_pair_detail(key[0], key[1], details)}")

    lines.append("")
    lines.append("**Justification chain (each event beats all below it in the order):**")
    for i, ev in enumerate(order):
        if i == len(order) - 1:
            break
        below = order[i + 1 :]
        wins_over = [o for o in below if o in beats[ev]]
        losses_to = [o for o in below if ev in beats[o]]
        if wins_over == below:
            beaten = ", ".join(wins_over)
            lines.append(f"  • {ev}: wins over {beaten}")
        elif wins_over:
            beaten = ", ".join(wins_over)
            if losses_to:
                not_beaten = ", ".join(losses_to)
                lines.append(
                    f"  • {ev}: wins over {beaten}; "
                    f"does NOT beat {not_beaten} (placed above by Copeland score {scores[ev]})"
                )
            else:
                lines.append(f"  • {ev}: wins over {beaten}")
        else:
            lines.append(
                f"  • {ev}: does not decisively beat all events below; "
                f"ranked by Copeland score ({scores[ev]} head-to-head wins)"
            )

    tied_pairs = [
        (ev_a, ev_b, count_a, count_b, ties, n_both)
        for ev_a, ev_b, count_a, count_b, ties, n_both in pairings
        if count_a == count_b
    ]
    if tied_pairs:
        lines.append("")
        lines.append("**Tied pairwise comparisons (no decisive winner):**")
        for ev_a, ev_b, count_a, count_b, ties, n_both in tied_pairs:
            extra = f"; ties in comparison: {ties}" if ties else ""
            lines.append(
                f"  • {ev_a} vs. {ev_b}: {count_a} each ({n_both} athletes in both{extra})"
            )

    if cycles:
        lines.append("")
        lines.append("**Non-transitive cycles (no single strict pecking order exists):**")
        shown: set[tuple[str, ...]] = set()
        for cycle in cycles:
            key = tuple(sorted(cycle))
            if key in shown:
                continue
            shown.add(key)
            a, b, c = cycle
            lines.append(
                f"  • {a} beats {b}, {b} beats {c}, {c} beats {a} "
                f"(Copeland scores used to break ambiguity)"
            )

    lines.append("")
    lines.append("**Copeland scores (number of other events beaten head-to-head):**")
    for ev in sorted(events, key=lambda e: (-scores[e], e)):
        lines.append(f"  • {ev}: {scores[ev]}")
    lines.append("")

    if excluded:
        lines.append(f"**Excluded pairings (fewer than {min_n} athletes in both):**")
        for ev_a, ev_b, _, _, _, n_both in excluded:
            lines.append(f"  • {ev_a} vs. {ev_b}: {n_both} athletes in both")
        lines.append("")

    return lines


def build_report(min_n: int) -> str:
    if min_n > 0:
        title = f"Event Pecking Orders (≥{min_n} Athletes in Both Events)"
        filename_note = (
            f"Only pairwise comparisons with at least {min_n} athletes "
            "competing in BOTH events are used."
        )
    else:
        title = "Event Pecking Orders from Pairwise Best-Event Comparisons"
        filename_note = "All pairwise comparisons included."

    out_lines = [
        title,
        "=" * len(title),
        "Outdoor track 2024–2026 combined (relay-inclusive datasets).",
        "",
        filename_note,
        "",
        "Method: For each qualifying event pair, count whose personal-best WA is",
        "higher. The event with more athletes as best event wins that pairing.",
        "Pecking order ranks events by head-to-head wins; when pairwise results",
        "are cyclic or incomplete, Copeland score (total head-to-head wins)",
        "breaks ties.",
        "",
    ]

    for discipline, folder in DISCIPLINES:
        out_lines.append(f"## {discipline}")
        out_lines.append("")
        for gender_label, gender_slug in GENDERS:
            for variant_name, suffix, variant_desc in VARIANTS:
                path = ROOT / folder / f"pairwise_best_event_counts_{gender_slug}_2024_2026{suffix}.txt"
                all_pairings = parse_pairwise_file(path)
                pairings, excluded = filter_pairings(all_pairings, min_n)
                out_lines.extend(
                    section_for_variant(
                        gender_label,
                        variant_name.replace("_", " ").title(),
                        variant_desc,
                        pairings,
                        excluded,
                        min_n,
                    )
                )

    return "\n".join(out_lines).rstrip() + "\n"


def main() -> None:
    full_path = ROOT / "event_pecking_orders.txt"
    full_path.write_text(build_report(min_n=0))
    print(f"Wrote {full_path}")

    min50_path = ROOT / "event_pecking_orders_min50.txt"
    min50_path.write_text(build_report(min_n=50))
    print(f"Wrote {min50_path}")


if __name__ == "__main__":
    main()

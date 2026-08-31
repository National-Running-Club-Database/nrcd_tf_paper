"""Scrapers for newly added NCAA D1 men's outdoor performance documents.

All schools here are NCAA Division I. Used by the D1_Only validation path.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pdfplumber

from preprocess import (
    FIELD_EVENTS,
    MEN_DOCS,
    cluster_lines,
    extract_pdf_text,
    make_row,
    parse_mark,
    words_by_column,
)

# Shared event aliases seen across D1 PDFs
EVENT_ALIASES: dict[str, str] = {
    "100m": "100m",
    "100": "100m",
    "100 meters": "100m",
    "100 meter dash": "100m",
    "100-meter dash": "100m",
    "200m": "200m",
    "200": "200m",
    "200 meters": "200m",
    "200 meter dash": "200m",
    "200-meter dash": "200m",
    "400m": "400m",
    "400": "400m",
    "400 meters": "400m",
    "400 meter dash": "400m",
    "400-meter dash": "400m",
    "400h": "400mH",
    "400mh": "400mH",
    "400 hurdles": "400mH",
    "400-meter hurdles": "400mH",
    "400 meter hurdles": "400mH",
    "110mh": "110mH",
    "110h": "110mH",
    "110 hurdles": "110mH",
    "110-meter hurdles": "110mH",
    "110 meter hurdles": "110mH",
    "800m": "800m",
    "800": "800m",
    "800 meters": "800m",
    "800 meter run": "800m",
    "800-meter run": "800m",
    "1500m": "1500m",
    "1,500m": "1500m",
    "1500": "1500m",
    "1,500": "1500m",
    "1500 meters": "1500m",
    "1,500 meters": "1500m",
    "1500 meter run": "1500m",
    "1,500 meter run": "1500m",
    "1500-meter run": "1500m",
    "1,500-meter run": "1500m",
    "mile": "Mile",
    "3000m": "3000m",
    "3,000m": "3000m",
    "3000": "3000m",
    "3,000": "3000m",
    "3000 meters": "3000m",
    "3,000 meters": "3000m",
    "3000 meter run": "3000m",
    "3,000 meter run": "3000m",
    "3000-meter run": "3000m",
    "3,000-meter run": "3000m",
    "3000m sc": "3000m SC",
    "3000m steeplechase": "3000m SC",
    "3,000m steeplechase": "3000m SC",
    "3000 steeplechase": "3000m SC",
    "3,000 steeplechase": "3000m SC",
    "3000-meter steeplechase": "3000m SC",
    "3,000-meter steeplechase": "3000m SC",
    "3,000 m steeple": "3000m SC",
    "5000m": "5000m",
    "5,000m": "5000m",
    "5000": "5000m",
    "5,000": "5000m",
    "5000 meters": "5000m",
    "5,000 meters": "5000m",
    "5000 meter run": "5000m",
    "5,000 meter run": "5000m",
    "5000-meter run": "5000m",
    "5,000-meter run": "5000m",
    "10000m": "10000m",
    "10,000m": "10000m",
    "10000": "10000m",
    "10,000": "10000m",
    "10000 meters": "10000m",
    "10,000 meters": "10000m",
    "10000 meter run": "10000m",
    "10,000 meter run": "10000m",
    "10000-meter run": "10000m",
    "10,000-meter run": "10000m",
    "hj": "HJ",
    "high jump": "HJ",
    "pv": "PV",
    "pole vault": "PV",
    "lj": "LJ",
    "long jump": "LJ",
    "tj": "TJ",
    "triple jump": "TJ",
    "sp": "SP",
    "shot put": "SP",
    "dt": "DT",
    "discus": "DT",
    "discus throw": "DT",
    "ht": "HT",
    "hammer": "HT",
    "hammer throw": "HT",
    "jt": "JT",
    "javelin": "JT",
    "javelin throw": "JT",
}


def _canon_event(text: str) -> str | None:
    key = re.sub(r"\s+", " ", text.strip().lower())
    key = key.replace("–", "-").replace("—", "-")
    key = re.sub(r"\s*\(.*?\)\s*", " ", key).strip()
    key = re.sub(r"\s*\|\s*.*$", "", key).strip()
    key = re.sub(r"\s*-\s*[pf]$", "", key).strip()  # prelim/final suffix
    key = re.sub(r"\s+", " ", key)
    return EVENT_ALIASES.get(key)


def _parse_mdy(text: str, default_year: int = 2026) -> date | None:
    text = text.strip()
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 100:
            yy += 2000
        try:
            return date(yy, mm, dd)
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", text)
    if m:
        try:
            return date(default_year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return None
    m = re.fullmatch(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s*(\d{4}))?",
        text,
        re.I,
    )
    if m:
        months = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        yy = int(m.group(3)) if m.group(3) else default_year
        try:
            return date(yy, months[m.group(1).lower()[:3]], int(m.group(2)))
        except ValueError:
            return None
    return None


def _imperial_to_metres(text: str) -> float | None:
    """Parse 15-1.75 or 139-7 style field marks to metres."""
    m = re.fullmatch(r"(\d+)-(\d+(?:\.\d+)?)", text.strip())
    if not m:
        return None
    feet = int(m.group(1))
    inches = float(m.group(2))
    return round(feet * 0.3048 + inches * 0.0254, 3)


def _clean_mark_token(raw: str, event: str) -> tuple[str, float | None]:
    raw = raw.strip().lstrip("a").rstrip("w!*^")
    raw = raw.replace(";", ":")
    # Troy typo 1.58.76 → 1:58.76
    if re.fullmatch(r"\d+\.\d{2}\.\d{2}", raw):
        a, b, c = raw.split(".")
        raw = f"{a}:{b}.{c}"
    if event in FIELD_EVENTS and re.fullmatch(r"\d+-\d+(?:\.\d+)?", raw):
        metres = _imperial_to_metres(raw)
        if metres is None:
            return raw, None
        return f"{metres:.2f}m", metres
    return parse_mark(event, raw)


def scrape_air_force(coeffs: dict) -> list[dict]:
    """Air Force (USAFA) meet-by-meet results — men's pages only.

    Source footer: GOAIRFORCEFALCONS.COM / @AF_TFXC.
    Pages 1–4 are men; pages 5–8 are women (excluded).
    """
    path = MEN_DOCS / "2026_Outdoor_Marks.pdf"
    rows: list[dict] = []
    name_re = re.compile(r"^([A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+)+)\s*\.{3,}")
    line_re = re.compile(
        r"(?P<md>\d{2}\.\d{2})\s*\|\s*(?P<meet>.+?)\.{2,}"
        r"(?P<event>[A-Za-z0-9][A-Za-z0-9 \-/]*?)\.{2,}"
        r"(?:(?P<place>\d+)\|\d+\s*)?\.{0,}"
        r"(?P<mark>a?\d+:\d{2}\.\d+|a?\d+\.\d+|\d+-\d+(?:\.\d+)?)",
        re.I,
    )

    with pdfplumber.open(path) as pdf:
        # Men only: first four pages
        text = "\n".join((pg.extract_text() or "") for pg in pdf.pages[:4])

    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        nm = name_re.match(line)
        if nm:
            current = re.sub(r"\s+", " ", nm.group(1).title())
            rest = line[nm.end() :]
            line = rest.strip()
            if not line:
                continue
        if current is None:
            continue
        for m in line_re.finditer(line):
            ev = _canon_event(m.group("event"))
            if not ev or ev.startswith("4x") or ev == "100mH":
                continue
            result_str, value = _clean_mark_token(m.group("mark"), ev)
            if value is None:
                continue
            comp_date = _parse_mdy(m.group("md"), 2026)
            if comp_date is None:
                continue
            meet = re.sub(r"\s+", " ", re.sub(r"\.{2,}", " ", m.group("meet")).strip())
            rows.append(
                make_row(
                    coeffs=coeffs,
                    athlete=current,
                    event=ev,
                    result_str=result_str,
                    mark_value=value,
                    indoor_outdoor="Outdoor",
                    competition_date=comp_date,
                    college="Air Force",
                    division="NCAA D1",
                    meet=meet,
                    source=path.name,
                    place=int(m.group("place")) if m.group("place") else None,
                )
            )
    return rows


def scrape_portland_state(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "2026_Performance_List.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    athlete_re = re.compile(r"^([A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)+)\s*$")
    # 800m (Open B) 1:54.44 63rd Bryan Clay Invitational 4/17/26
    result_re = re.compile(
        r"^(?P<event>.+?)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)\s+"
        r"(?P<place>\d+(?:st|nd|rd|th)|DNF|DNS|DQ|NH|FOUL|-)\s+"
        r"(?P<meet>.+?)\s+"
        r"(?P<md>\d{1,2}/\d{1,2}/\d{2})\s*$",
        re.I,
    )
    current: str | None = None
    outdoor = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.search(r"2026 Outdoor Performance List", line, re.I):
            outdoor = True
            continue
        if re.search(r"2026 Indoor Performance List", line, re.I):
            outdoor = False
            continue
        if athlete_re.match(line) and "Performance" not in line and "Results" not in line:
            # skip bio lines like "Mid-Distances | RS-SR | ..."
            if "|" in line:
                continue
            current = line
            continue
        if not outdoor or current is None:
            continue
        if line.lower().startswith("event "):
            continue
        rm = result_re.match(line)
        if not rm:
            continue
        place_tok = rm.group("place")
        if place_tok.upper() in {"DNF", "DNS", "DQ", "NH", "FOUL", "-"}:
            continue
        ev = _canon_event(rm.group("event"))
        if not ev or ev.startswith("4x") or "relay" in rm.group("event").lower():
            continue
        result_str, value = _clean_mark_token(rm.group("mark"), ev)
        if value is None:
            continue
        comp_date = _parse_mdy(rm.group("md"), 2026)
        if comp_date is None:
            continue
        place = int(re.match(r"\d+", place_tok).group(0))
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=current,
                event=ev,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=comp_date,
                college="Portland State",
                division="NCAA D1",
                meet=rm.group("meet").strip(),
                source=path.name,
                place=place,
            )
        )
    return rows


def scrape_troy(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "2026_Troy_Track_and_Field_Performance_List.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    # Two-column layout: process full text with athlete/event headers
    athlete_re = re.compile(r"^([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)\s*$")
    event_re = re.compile(
        r"^(100m|200m|400m|800m|1500m|5000m|10,000m|10000m|400H|110H|HJ|PV|LJ|TJ|SP|DT|HT|JT|"
        r"High Jump|Pole Vault|Long Jump|Triple Jump|Shot Put|Discus|Hammer|Javelin)\s*$",
        re.I,
    )
    result_re = re.compile(
        r"^(?P<mark>\d+:\d{2}\.\d+|\d+\.\d{2}\.\d{2}|\d+\.\d+|\d+-\d+(?:\.\d+)?)\s+"
        r"(?P<meet>.+?)\s+"
        r"(?P<md>\d{1,2}/\d{1,2}/\d{2})\s+"
        r"(?P<place>\d+(?:st|nd|rd|th)(?:\s*\([PF]\))?|DNS|DQ|DNF|---)\s*$",
        re.I,
    )
    current_athlete: str | None = None
    current_event: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("2026 TROY") or line in {"Time Meet Date Finish"}:
            continue
        if athlete_re.match(line) and _canon_event(line) is None:
            current_athlete = line
            current_event = None
            continue
        em = event_re.match(line)
        if em:
            current_event = _canon_event(em.group(1))
            continue
        if current_athlete is None or current_event is None:
            continue
        if current_event.startswith("4x"):
            continue
        rm = result_re.match(line)
        if not rm:
            continue
        if rm.group("place").upper() in {"DNS", "DQ", "DNF", "---"}:
            continue
        result_str, value = _clean_mark_token(rm.group("mark"), current_event)
        if value is None:
            continue
        comp_date = _parse_mdy(rm.group("md"), 2026)
        if comp_date is None:
            continue
        place_m = re.match(r"\d+", rm.group("place"))
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=current_athlete,
                event=current_event,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=comp_date,
                college="Troy",
                division="NCAA D1",
                meet=rm.group("meet").strip(),
                source=path.name,
                place=int(place_m.group(0)) if place_m else None,
            )
        )
    return rows


def scrape_providence(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "Accessible_2025-26_Men_s_Outdoor_Track_Results.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    # Meet headers include a date line; results: Event Place Athlete Time
    meet_date: date | None = None
    meet_name = "Providence meet"
    result_re = re.compile(
        r"^(?P<event>\d[\d,]*\s*M(?:eters)?|\d[\d,]*\s*M\s*Steeple|High Jump|Long Jump|Triple Jump|"
        r"Pole Vault|Shot Put|Discus|Hammer|Javelin|100 Meters|200 Meters|400 Meters|800 Meters|"
        r"1,500 Meters|3,000 Meters|5,000 Meters|10,000 Meters|3,000 M Steeple)\s+"
        r"(?P<place>\d+(?:st|nd|rd|th))\s+"
        r"(?P<athlete>[A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+)+)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)\s*$",
        re.I,
    )
    date_line_re = re.compile(
        r"(?:Thursday|Friday|Saturday|Sunday|Monday|Tuesday|Wednesday)?,?\s*"
        r"(?P<mon>January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(?P<d1>\d{1,2})(?:\s*-\s*(?:Saturday|Sunday|Monday|Tuesday|Wednesday|Thursday|Friday)?\s*"
        r"(?P<mon2>January|February|March|April|May|June|July|August|September|October|November|December)?\s*"
        r"(?P<d2>\d{1,2}))?",
        re.I,
    )
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Meet title lines (short, not a result)
        is_meet_title = (
            not result_re.match(line)
            and not line.startswith("Event ")
            and not line.startswith("Team Score")
            and not line.startswith("@ ")
            and "Results" not in line
            and len(line) < 60
            and bool(re.search(r"Relays|Invitational|Championship|Classic|Challenge", line, re.I))
            and not re.search(r"\d+\.\d+", line)
        )
        if is_meet_title:
            meet_name = line
            continue
        dm = date_line_re.search(line)
        if dm and not result_re.match(line):
            mon = months[dm.group("mon").lower()]
            day = int(dm.group("d2") or dm.group("d1"))
            try:
                meet_date = date(2026, mon, day)
            except ValueError:
                meet_date = None
            continue
        rm = result_re.match(line)
        if not rm or meet_date is None:
            continue
        ev = _canon_event(rm.group("event"))
        if not ev or ev.startswith("4x"):
            continue
        result_str, value = _clean_mark_token(rm.group("mark"), ev)
        if value is None:
            continue
        place = int(re.match(r"\d+", rm.group("place")).group(0))
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=rm.group("athlete").strip(),
                event=ev,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=meet_date,
                college="Providence",
                division="NCAA D1",
                meet=meet_name,
                source=path.name,
                place=place,
            )
        )
    return rows


def scrape_indiana_state(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "Outdoor_Performance_List.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    event_header_re = re.compile(
        r"^(100 METERS|200 METERS|400 METERS|800 METERS|1500 METERS|1,500 METERS|"
        r"5000 METERS|5,000 METERS|10000 METERS|10,000 METERS|"
        r"110 METER HURDLES|400 METER HURDLES|HIGH JUMP|POLE VAULT|LONG JUMP|"
        r"TRIPLE JUMP|SHOT PUT|DISCUS|HAMMER|JAVELIN)\b",
        re.I,
    )
    # Full row with athlete, or continuation without name
    full_re = re.compile(
        r"^(?P<athlete>[A-Za-z][A-Za-z'\.\-]+(?:\s+[A-Za-z][A-Za-z'\.\-]+)+)\s+"
        r"(?P<md>\d{1,2}/\d{1,2}/\d{2})\s+"
        r"(?P<meet>.+?)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+w?|\d+\.\d+m?)\s*$"
    )
    cont_re = re.compile(
        r"^(?P<md>\d{1,2}/\d{1,2}/\d{2})\s+"
        r"(?P<meet>.+?)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+w?|\d+\.\d+m?)\s*$"
    )
    current_event: str | None = None
    last_athlete: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hm = event_header_re.match(line)
        if hm:
            current_event = _canon_event(hm.group(1))
            last_athlete = None
            continue
        if current_event is None or current_event.startswith("4x"):
            continue
        if line.startswith("Name ") or "Record" in line:
            continue
        fm = full_re.match(line)
        if fm:
            last_athlete = fm.group("athlete").strip()
            mark_raw = fm.group("mark").rstrip("w")
            result_str, value = _clean_mark_token(mark_raw, current_event)
            if value is None:
                continue
            comp_date = _parse_mdy(fm.group("md"), 2026)
            if comp_date is None:
                continue
            rows.append(
                make_row(
                    coeffs=coeffs,
                    athlete=last_athlete,
                    event=current_event,
                    result_str=result_str,
                    mark_value=value,
                    indoor_outdoor="Outdoor",
                    competition_date=comp_date,
                    college="Indiana State",
                    division="NCAA D1",
                    meet=fm.group("meet").strip(),
                    source=path.name,
                )
            )
            continue
        cm = cont_re.match(line)
        if cm and last_athlete:
            mark_raw = cm.group("mark").rstrip("w")
            result_str, value = _clean_mark_token(mark_raw, current_event)
            if value is None:
                continue
            comp_date = _parse_mdy(cm.group("md"), 2026)
            if comp_date is None:
                continue
            rows.append(
                make_row(
                    coeffs=coeffs,
                    athlete=last_athlete,
                    event=current_event,
                    result_str=result_str,
                    mark_value=value,
                    indoor_outdoor="Outdoor",
                    competition_date=comp_date,
                    college="Indiana State",
                    division="NCAA D1",
                    meet=cm.group("meet").strip(),
                    source=path.name,
                )
            )
    return rows


def scrape_utep(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "2026_outdoor_peformance_list.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    athlete_re = re.compile(r"^([A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+)+)\s*$")
    event_re = re.compile(
        r"^(100M|200M|400M|800M|1500M|5000M|10000M|10,000M|110H|400H|HJ|PV|LJ|TJ|SP|DT|HT|JT|"
        r"HIGH JUMP|POLE VAULT|LONG JUMP|TRIPLE JUMP|SHOT PUT|DISCUS|HAMMER|JAVELIN|"
        r"3000M SC|STEEPLE)\s*$",
        re.I,
    )
    # Mar. 21 __Meet ______________10.86 (0.1) ______________7th
    result_re = re.compile(
        r"^(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(?P<day>\d{1,2})\s+"
        r"_+(?P<meet>.+?)_+\s*"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?(?:\s*\([^)]*\))?|\d+\.\d+)\s*"
        r"_+\s*(?P<place>\d+(?:st|nd|rd|th)|prelims|finals)?",
        re.I,
    )
    current_athlete: str | None = None
    current_event: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("MEN") or line.startswith("LAST UPDATED") or "SCHOOL RECORD" in line:
            continue
        if athlete_re.match(line) and _canon_event(line) is None and "METER" not in line.upper():
            current_athlete = line.title()
            current_event = None
            continue
        em = event_re.match(line)
        if em:
            current_event = _canon_event(em.group(1))
            continue
        if current_athlete is None or current_event is None:
            continue
        rm = result_re.match(line)
        if not rm:
            continue
        mark_raw = re.sub(r"\s*\([^)]*\)\s*", "", rm.group("mark")).strip()
        result_str, value = _clean_mark_token(mark_raw, current_event)
        if value is None:
            continue
        comp_date = _parse_mdy(f"{rm.group('mon')} {rm.group('day')}", 2026)
        if comp_date is None:
            continue
        meet = re.sub(r"_+", " ", rm.group("meet")).strip()
        place = None
        if rm.group("place") and rm.group("place").isdigit() is False:
            pm = re.match(r"\d+", rm.group("place") or "")
            place = int(pm.group(0)) if pm else None
        elif rm.group("place"):
            place = int(re.match(r"\d+", rm.group("place")).group(0))
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=current_athlete,
                event=current_event,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=comp_date,
                college="UTEP",
                division="NCAA D1",
                meet=meet,
                source=path.name,
                place=place,
            )
        )
    return rows


def scrape_utah_state_supplementary(coeffs: dict) -> list[dict]:
    """Athlete lists without calendar dates → season-best rows."""
    path = MEN_DOCS / "2026_Outdoor_Performance_List_Utah_State.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    # 5,000 Meters Stanford Invitational 14:29.20 21st
    result_re = re.compile(
        r"^(?P<event>\d[\d,]*[- ]?Meters?(?:\s+Steeplechase)?|Mile|High Jump|Pole Vault|Long Jump|"
        r"Triple Jump|Shot Put|Discus|Hammer|Javelin|110-Meter Hurdles|400-Meter Hurdles|"
        r"100 Meters|200 Meters|400 Meters|800 Meters)\s+"
        r"(?P<meet>.+?)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m(?:/\d+-\d+(?:\.\d+)?)?|\d+\.\d+)\s+"
        r"(?P<place>\d+(?:st|nd|rd|th)|NH|DNF|DNS)",
        re.I,
    )
    athlete_re = re.compile(r"^([A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-\.]+)+)(?:\s+[A-Z][A-Z'\-]+)*\s*$")
    best: dict[tuple[str, str], tuple[float, str]] = {}
    current: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("MEN") or line.startswith("Event Meet") or line == "®":
            continue
        # Athlete headers are ALL CAPS names without digits
        if athlete_re.match(line) and not re.search(r"\d", line) and "METER" not in line:
            # May be two names side by side — take left token group
            parts = re.split(r"\s{2,}", line)
            current = parts[0].title()
            continue
        if current is None:
            continue
        for m in result_re.finditer(line):
            if m.group("place").upper() in {"NH", "DNF", "DNS"}:
                continue
            ev = _canon_event(m.group("event"))
            if not ev or ev.startswith("4x") or "relay" in m.group("event").lower():
                continue
            mark_raw = m.group("mark")
            if "/" in mark_raw and "m/" in mark_raw.lower():
                mark_raw = mark_raw.split("/")[0]
            result_str, value = _clean_mark_token(mark_raw, ev)
            if value is None:
                continue
            key = (current, ev)
            prev = best.get(key)
            better = prev is None or (
                value > prev[0] if ev in FIELD_EVENTS else value < prev[0]
            )
            if better:
                best[key] = (value, result_str)
    for (athlete, event), (value, result_str) in best.items():
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=athlete,
                event=event,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=None,
                college="Utah State",
                division="NCAA D1",
                meet="season best (undated performance list)",
                source=path.name,
                season_year=2026,
                source_role="supplementary_season_pb",
            )
        )
    return rows


def scrape_marquette(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "MUTF_Outdoor_Performance_List_FINAL.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    event_header_re = re.compile(
        r"^(100 METER|200 METER|400 METER|800 METER|1500 METER|1,500 METER|"
        r"5000 METER|5,000 METER|10000 METER|10,000 METER|"
        r"110 METER HURDLE|400 METER HURDLE|HIGH JUMP|POLE VAULT|LONG JUMP|"
        r"TRIPLE JUMP|SHOT PUT|DISCUS|HAMMER|JAVELIN|4 x )",
        re.I,
    )
    # 10.xx Name Month Day Meet   OR   1:xx.xx Name ...
    result_re = re.compile(
        r"^(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+(?:\s*/\s*\d+-\d+(?:\.\d+)?)?)\s+"
        r"(?:\([^)]+\)\s+)?"
        r"(?P<athlete>[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)\s+"
        r"(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
        r"(?P<day>\d{1,2})\s+"
        r"(?P<meet>.+)$",
        re.I,
    )
    current_event: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hm = event_header_re.match(line)
        if hm:
            if line.upper().startswith("4 X"):
                current_event = None
            else:
                current_event = _canon_event(hm.group(0) + ("S" if "HURDLE" not in hm.group(0).upper() else "S"))
                # Fix headers like "100 METER" → need "100 METERS"
                if current_event is None:
                    current_event = _canon_event(hm.group(0).rstrip() + "S")
            continue
        if current_event is None or current_event.startswith("4x"):
            continue
        if line.startswith("School:") or line.startswith("Freshman:"):
            continue
        rm = result_re.match(line)
        if not rm:
            continue
        mark_raw = rm.group("mark").split("/")[0].strip()
        result_str, value = _clean_mark_token(mark_raw, current_event)
        if value is None:
            continue
        comp_date = _parse_mdy(f"{rm.group('mon')} {rm.group('day')}", 2026)
        if comp_date is None:
            continue
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=rm.group("athlete").strip(),
                event=current_event,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=comp_date,
                college="Marquette",
                division="NCAA D1",
                meet=rm.group("meet").strip(),
                source=path.name,
            )
        )
    return rows


def scrape_montana_top5(coeffs: dict) -> list[dict]:
    """Men's Top-5 lists with month/day only (year=2026)."""
    path = MEN_DOCS / "2026_outdoor_performance_list.pdf"
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[1].extract_text() or ""
    rows: list[dict] = []
    line_re = re.compile(
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+|\d+-\d+(?:\.\d+)?)"
        r"(?:\s*\([^)]*\))?"
        r"[\.…]+\s*"
        r"(?P<athlete>[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)"
        r"[\.…]+\s*"
        r"(?P<md>\d{1,2}/\d{1,2})"
    )
    event_re = re.compile(
        r"(100 Meters|200 Meters|400 Meters|800 Meters|1,500 Meters|5,000 Meters|"
        r"110-Meter Hurdles|400-Meter Hurdles|3,000-Meter Steeplechase|"
        r"High Jump|Pole Vault|Long Jump|Triple Jump|Shot Put|Discus|Hammer|Javelin)",
        re.I,
    )
    # Walk the page left-to-right by finding event headers and consuming following marks
    # until the next event header.
    matches = list(event_re.finditer(text))
    for idx, em in enumerate(matches):
        ev = _canon_event(em.group(1))
        if not ev or ev.startswith("4x"):
            continue
        start = em.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end]
        for m in line_re.finditer(body):
            result_str, value = _clean_mark_token(m.group("mark"), ev)
            if value is None:
                continue
            comp_date = _parse_mdy(m.group("md") + "/26", 2026)
            if comp_date is None:
                continue
            rows.append(
                make_row(
                    coeffs=coeffs,
                    athlete=m.group("athlete").strip(),
                    event=ev,
                    result_str=result_str,
                    mark_value=value,
                    indoor_outdoor="Outdoor",
                    competition_date=comp_date,
                    college="Montana",
                    division="NCAA D1",
                    meet="Montana top-5 list",
                    source=path.name,
                )
            )
    return rows


NEW_D1_PRIMARY = [
    ("Air Force", scrape_air_force),
    ("Portland State", scrape_portland_state),
    ("Troy", scrape_troy),
    ("Providence", scrape_providence),
    ("Indiana State", scrape_indiana_state),
    ("UTEP", scrape_utep),
    ("Marquette", scrape_marquette),
    ("Montana", scrape_montana_top5),
]

NEW_D1_SUPPLEMENTARY = [
    ("Utah State (suppl.)", scrape_utah_state_supplementary),
]

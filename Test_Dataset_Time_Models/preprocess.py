"""Build the men's test dataset for time-model validation.

Scrapes competition results from documents/men, scores them with World
Athletics 2025 outdoor tables, and adds feature columns:

  • bal_spec / best_event (and related routing flags)
  • chronological markers (result_order_in_season, season_day_index)
  • starter tolerance bands for predicted-vs-actual checks

Primary sources have per-result competition dates (season-PB + chronological).
Supplementary undated season lists / HTML matrices are included as season-best
rows only (Source_Role = supplementary_season_pb); chronological validation
ignores them because season-PB is the stronger protocol.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scoring.api import score as multi_score  # noqa: E402
from scoring.marks import parse_mark as scoring_parse_mark  # noqa: E402
from scoring.marks import parse_time_to_seconds as scoring_parse_time  # noqa: E402
from scoring.wa import load_coefficients as scoring_load_coefficients  # noqa: E402
from scoring.wa import wa_points as scoring_wa_points  # noqa: E402

MEN_DOCS = ROOT / "documents" / "men"
OUT_DIR = ROOT / "output"
COEFF_PATH = PROJECT_ROOT / "scoring" / "data" / "coefficients-2025.json"
SPREAD_THRESHOLD = 50.0

# Event groups used for bal_spec / best_event (aligned with outdoor time models)
EVENT_GROUPS = {
    "Sprints": {"100m", "200m", "400m"},
    "Distance": {"800m", "1500m", "3000m", "3000m SC", "5000m", "10000m"},
    "Hurdles": {"110mH", "400mH"},
    "Jumps": {"HJ", "PV", "LJ", "TJ"},
    "Throws": {"SP", "DT", "HT", "JT"},
}

# Starter absolute-time tolerances (seconds) for predicted-vs-actual checks.
# Field events use metres. These are placeholders until pooled-model CVs are wired in.
TOLERANCE_SECONDS = {
    "100m": 0.15,
    "200m": 0.30,
    "400m": 0.80,
    "800m": 1.50,
    "1500m": 3.00,
    "3000m": 6.00,
    "3000m SC": 8.00,
    "5000m": 10.00,
    "10000m": 20.00,
    "110mH": 0.25,
    "400mH": 1.00,
}
TOLERANCE_METRES = {
    "HJ": 0.05,
    "PV": 0.10,
    "LJ": 0.15,
    "TJ": 0.25,
    "SP": 0.40,
    "DT": 1.50,
    "HT": 1.50,
    "JT": 1.50,
}

SIUE_EVENT_MAP = {
    "100 Meters": "100m",
    "200 Meters": "200m",
    "400 Meters": "400m",
    "800 Meters": "800m",
    "1500 Meters": "1500m",
    "3000 Meters": "3000m",
    "3000 Steeplechase": "3000m SC",
    "5000 Meters": "5000m",
    "110 Hurdles": "110mH",
    "400 Hurdles": "400mH",
    "High Jump": "HJ",
    "Pole Vault": "PV",
    "Long Jump": "LJ",
    "Triple Jump": "TJ",
    "Shot Put": "SP",
    "Discus": "DT",
    "Hammer": "HT",
    "Javelin": "JT",
    "4 x 100 Relay": "4x100m",
    "4 x 400 Relay": "4x400m",
}

NF_EVENT_MAP = {
    "100 METERS": "100m",
    "200 METERS": "200m",
    "400 METERS": "400m",
    "800 METERS": "800m",
    "1500 METERS": "1500m",
    "5000 METERS": "5000m",
    "10,000 METERS": "10000m",
    "110 METER HURDLES": "110mH",
    "400 METER HURDLES": "400mH",
    "3000 METER STEEPLECHASE": "3000m SC",
    "HIGH JUMP": "HJ",
    "POLE VAULT": "PV",
    "LONG JUMP": "LJ",
    "TRIPLE JUMP": "TJ",
    "SHOT PUT": "SP",
    "DISCUS": "DT",
    "HAMMER": "HT",
    "JAVELIN": "JT",
    "4X100M RELAY": "4x100m",
    "4X400M RELAY": "4x400m",
}

DEPAUL_EVENT_MAP = {
    "100": "100m",
    "200": "200m",
    "400": "400m",
    "800": "800m",
    "1500": "1500m",
    "3000": "3000m",
    "5000": "5000m",
    "10000": "10000m",
    "3000SC": "3000m SC",
    "110H": "110mH",
    "400H": "400mH",
    "HJ": "HJ",
    "PV": "PV",
    "LJ": "LJ",
    "TJ": "TJ",
    "SP": "SP",
    "DT": "DT",
    "HT": "HT",
    "JT": "JT",
    "4x100": "4x100m",
    "4x400": "4x400m",
}

WA_EVENT_KEYS = {
    "100m": "100m",
    "200m": "200m",
    "400m": "400m",
    "800m": "800m",
    "1500m": "1500m",
    "3000m": "3000m",
    "3000m SC": "3000m SC",
    "5000m": "5000m",
    "10000m": "10000m",
    "110mH": "110mH",
    "400mH": "400mH",
    "HJ": "HJ",
    "PV": "PV",
    "LJ": "LJ",
    "TJ": "TJ",
    "SP": "SP",
    "DT": "DT",
    "HT": "HT",
    "JT": "JT",
    "4x100m": "4x100m",
    "4x400m": "4x400m",
}

MONTHS = {
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

TRACK_EVENTS = {
    "100m",
    "200m",
    "400m",
    "800m",
    "1500m",
    "3000m",
    "3000m SC",
    "5000m",
    "10000m",
    "110mH",
    "400mH",
    "4x100m",
    "4x400m",
}
FIELD_EVENTS = {"HJ", "PV", "LJ", "TJ", "SP", "DT", "HT", "JT"}


def load_coefficients() -> dict:
    return scoring_load_coefficients()


def wa_points(coeffs: dict, gender: str, event: str, mark_value: float) -> int | None:
    return scoring_wa_points(coeffs, gender, event, mark_value)


def parse_time_to_seconds(text: str) -> float | None:
    return scoring_parse_time(text)


def parse_mark(event: str, raw: str) -> tuple[str, float | None]:
    display, value = scoring_parse_mark(event, raw)
    return display or raw, value


def parse_siue_date(text: str) -> date | None:
    m = re.search(
        r"\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s*(\d{4})\)",
        text,
        re.I,
    )
    if not m:
        return None
    return date(int(m.group(3)), MONTHS[m.group(1).lower()[:3]], int(m.group(2)))


def parse_nf_date(text: str, year: int = 2024) -> date | None:
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2})", text, re.I)
    if not m:
        return None
    return date(year, MONTHS[m.group(1).lower()[:3]], int(m.group(2)))


def parse_depaul_date(text: str) -> date | None:
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2})", text)
    if not m:
        return None
    yy = int(m.group(3))
    year = 2000 + yy if yy < 70 else 1900 + yy
    return date(year, int(m.group(1)), int(m.group(2)))


def extract_pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


def scrape_siue(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / "2026_SIUE_Mens_TF_Outdoor_Performance_List.pdf"
    text = extract_pdf_text(path)
    rows: list[dict] = []
    current_event: str | None = None
    header_re = re.compile(r"^(" + "|".join(map(re.escape, SIUE_EVENT_MAP)) + r")\s*$")
    # Name Year Mark Meet...(Date) [Wind]
    row_re = re.compile(
        r"^(?P<athlete>.+?)\s+(?P<year>FR|SO|JR|SR|GR)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)\s+"
        r"(?P<meet>.+?\((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}\))"
        r"(?:\s+(?P<wind>-?\d+\.\d+))?$",
        re.I,
    )
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        hm = header_re.match(line)
        if hm:
            current_event = SIUE_EVENT_MAP[hm.group(1)]
            continue
        if line.startswith("Athlete ") or current_event is None:
            continue
        if current_event.startswith("4x"):
            continue  # skip relays for individual feature work
        rm = row_re.match(line)
        if not rm:
            continue
        result_str, value = parse_mark(current_event, rm.group("mark"))
        if value is None:
            continue
        comp_date = parse_siue_date(rm.group("meet"))
        if comp_date is None:
            continue
        meet = re.sub(r"\s*\([^)]*\)\s*$", "", rm.group("meet")).strip()
        rows.append(
            make_row(
                coeffs=coeffs,
                athlete=rm.group("athlete").strip(),
                event=current_event,
                result_str=result_str,
                mark_value=value,
                indoor_outdoor="Outdoor",
                competition_date=comp_date,
                college="SIUE",
                division="NCAA D1",
                meet=meet,
                source=path.name,
                wind=rm.group("wind"),
            )
        )
    return rows


def words_by_column(page, mid_x: float | None = None) -> tuple[list, list]:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    if not words:
        return [], []
    if mid_x is None:
        mid_x = (min(w["x0"] for w in words) + max(w["x1"] for w in words)) / 2
    left = [w for w in words if w["x0"] < mid_x]
    right = [w for w in words if w["x0"] >= mid_x]
    return left, right


def cluster_lines(words: list, y_tol: float = 3.0) -> list[str]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    lines: list[list] = []
    for w in words:
        if not lines or abs(w["top"] - lines[-1][0]["top"]) > y_tol:
            lines.append([w])
        else:
            lines[-1].append(w)
    out = []
    for line_words in lines:
        line_words = sorted(line_words, key=lambda w: w["x0"])
        out.append(" ".join(w["text"] for w in line_words))
    return out


NF_MEETS = [
    "ASUN Championships",
    "Bob Hayes Invitational",
    "East Coast Relays",
    "Florida Relays",
    "Georgia Tech Invite",
    "Georgie Tech Invite",  # typo in source PDF
    "Georgia Tech Invitational",
    "Tom Jones Memorial",
]


def scrape_north_florida(coeffs: dict) -> list[dict]:
    """Parse NF performances-by-event pages using left/right column word clusters."""
    path = MEN_DOCS / "2024_North_Florida_MTR_Outdoor_Results.pdf"
    rows: list[dict] = []
    event_header_re = re.compile(
        r"^(" + "|".join(sorted(map(re.escape, NF_EVENT_MAP), key=len, reverse=True)) + r")\s*$"
    )
    result_re = re.compile(
        r"^(?P<mark>(?:\d+:)?\d+(?:\.\d+)?(?:m)?|\d+\.\d{2}\.\d{2})\s+"
        r"(?P<body>.+?)\s+"
        r"(?P<md>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2})\s+"
        r"(?P<finish>\d+)\s*$",
        re.I,
    )
    meets_sorted = sorted(NF_MEETS, key=len, reverse=True)

    def split_athlete_meet(body: str) -> tuple[str, str] | None:
        body = re.sub(r"\s*\((?:P|\*)\)\s*", " ", body).strip()
        body = re.sub(r"\s+", " ", body)
        for meet in meets_sorted:
            # allow trailing school-record asterisk after meet
            if body.endswith(meet) or body.endswith(meet + "*"):
                athlete = body[: -len(meet)].rstrip(" *")
                athlete = athlete.strip()
                if athlete:
                    return athlete, meet
            # meet may sit in the middle if body has noise; prefer endswith
        # fallback: first two title-case tokens = athlete
        toks = body.split()
        if len(toks) >= 3:
            if toks[1].lower() in {"el", "de", "la", "le", "van", "von"} and len(toks) >= 4:
                return " ".join(toks[:3]), " ".join(toks[3:])
            return " ".join(toks[:2]), " ".join(toks[2:])
        return None

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:3]:  # by-event pages
            left_w, right_w = words_by_column(page)
            for col_words in (left_w, right_w):
                current_event: str | None = None
                for line in cluster_lines(col_words):
                    line = line.strip()
                    if not line or line.startswith("Time Name") or line.startswith("Mark Name"):
                        continue
                    hm = event_header_re.match(line)
                    if hm:
                        current_event = NF_EVENT_MAP[hm.group(1)]
                        continue
                    for ename, ecode in NF_EVENT_MAP.items():
                        if line.upper().startswith(ename):
                            current_event = ecode
                            line = line[len(ename) :].strip()
                            break
                    if current_event is None or current_event.startswith("4x"):
                        continue
                    rm = result_re.match(line)
                    if not rm:
                        continue
                    split = split_athlete_meet(rm.group("body"))
                    if not split:
                        continue
                    athlete, meet = split
                    result_str, value = parse_mark(current_event, rm.group("mark"))
                    if value is None:
                        continue
                    comp_date = parse_nf_date(rm.group("md"), year=2024)
                    if comp_date is None:
                        continue
                    rows.append(
                        make_row(
                            coeffs=coeffs,
                            athlete=athlete,
                            event=current_event,
                            result_str=result_str,
                            mark_value=value,
                            indoor_outdoor="Outdoor",
                            competition_date=comp_date,
                            college="North Florida",
                            division="NCAA D1",
                            meet=meet,
                            source=path.name,
                            place=int(rm.group("finish")),
                        )
                    )
    return rows


def scrape_depaul(coeffs: dict, *, men_only: bool = True) -> list[dict]:
    """TF_2025_Results.pdf contains Men's pages then Women's pages; keep Men for this folder."""
    path = MEN_DOCS / "TF_2025_Results.pdf"
    rows: list[dict] = []
    athlete_re = re.compile(
        r"^(?P<name>[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)\s+\((?P<year>Fr|So|Jr|Sr|Gr)\.?\)\s*$"
    )
    event_re = re.compile(
        r"^(?P<ev>" + "|".join(map(re.escape, DEPAUL_EVENT_MAP)) + r")\s*-\s*Career Best:",
        re.I,
    )
    result_re = re.compile(
        r"^(?P<meet>.+?)\s+"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)\s+"
        r"(?P<place>\d+(?:st|nd|rd|th)(?:\([PF]\))?)\s+"
        r"(?P<md>\d{1,2}/\d{1,2}/\d{2})\s*$",
        re.I,
    )

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if re.search(r"Women.?s Outdoor", page_text, re.I):
                if men_only:
                    continue
                page_gender = "Women"
            elif re.search(r"Men.?s Outdoor", page_text, re.I):
                page_gender = "Men"
            else:
                continue

            left_w, right_w = words_by_column(page)
            for col_words in (left_w, right_w):
                athlete: str | None = None
                current_event: str | None = None
                for line in cluster_lines(col_words):
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("DEPAUL")
                        or "Outdoor T&F Results" in line
                        or line.startswith("Updated")
                        or line.startswith("www.")
                    ):
                        continue
                    if re.fullmatch(r"\d+(\s+\d+)?", line):
                        continue
                    am = athlete_re.match(line)
                    if am:
                        athlete = am.group("name").strip()
                        current_event = None
                        continue
                    em = event_re.match(line)
                    if em and athlete:
                        current_event = DEPAUL_EVENT_MAP[em.group("ev")]
                        continue
                    if athlete is None or current_event is None:
                        continue
                    rm = result_re.match(line)
                    if not rm:
                        continue
                    result_str, value = parse_mark(current_event, rm.group("mark"))
                    if value is None:
                        continue
                    comp_date = parse_depaul_date(rm.group("md"))
                    if comp_date is None:
                        continue
                    place_num = int(re.match(r"(\d+)", rm.group("place")).group(1))
                    row = make_row(
                        coeffs=coeffs,
                        athlete=athlete,
                        event=current_event,
                        result_str=result_str,
                        mark_value=value,
                        indoor_outdoor="Outdoor",
                        competition_date=comp_date,
                        college="DePaul",
                        division="NCAA D1",
                        meet=rm.group("meet").strip(),
                        source=path.name,
                        place=place_num,
                    )
                    row["Gender"] = page_gender
                    rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Supplementary season-PB sources (undated matrices / performance lists)
# ---------------------------------------------------------------------------

SUPP_HTML_EVENT_MAP = {
    "100-meter dash": "100m",
    "100 meter dash": "100m",
    "100m dash": "100m",
    "100 meters": "100m",
    "200-meter dash": "200m",
    "200 meter dash": "200m",
    "200m dash": "200m",
    "200 meters": "200m",
    "400-meter dash": "400m",
    "400 meter dash": "400m",
    "400m dash": "400m",
    "400 meters": "400m",
    "800-meter run": "800m",
    "800 meter run": "800m",
    "800m run": "800m",
    "800 meters": "800m",
    "1500-meter run": "1500m",
    "1,500-meter run": "1500m",
    "1500 meter run": "1500m",
    "1500m run": "1500m",
    "1500 meters": "1500m",
    "1,500 meters": "1500m",
    "3000-meter run": "3000m",
    "3,000-meter run": "3000m",
    "3000 meters": "3000m",
    "3,000 meters": "3000m",
    "3000-meter steeplechase": "3000m SC",
    "3,000-meter steeplechase": "3000m SC",
    "3000m steeplechase": "3000m SC",
    "5000-meter run": "5000m",
    "5,000-meter run": "5000m",
    "5000 meters": "5000m",
    "5,000 meters": "5000m",
    "10000-meter run": "10000m",
    "10,000-meter run": "10000m",
    "10000 meters": "10000m",
    "10,000 meters": "10000m",
    "110-meter hurdles": "110mH",
    "110 meter hurdles": "110mH",
    "110m hurdles": "110mH",
    "400-meter hurdles": "400mH",
    "400 meter hurdles": "400mH",
    "400m hurdles": "400mH",
    "high jump": "HJ",
    "pole vault": "PV",
    "long jump": "LJ",
    "triple jump": "TJ",
    "shot put": "SP",
    "discus": "DT",
    "discus throw": "DT",
    "hammer": "HT",
    "hammer throw": "HT",
    "javelin": "JT",
    "javelin throw": "JT",
}

USI_EVENT_MAP = {
    "800 meters": "800m",
    "1,500 meters": "1500m",
    "1500 meters": "1500m",
    "3,000 meters": "3000m",
    "3000 meters": "3000m",
    "3,000-meter steeplechase": "3000m SC",
    "3000-meter steeplechase": "3000m SC",
    "5,000 meters": "5000m",
    "5000 meters": "5000m",
    "10,000 meters": "10000m",
    "10000 meters": "10000m",
}

KEISER_EVENT_PATTERNS = [
    (re.compile(r"100m\s*Dash", re.I), "100m"),
    (re.compile(r"200m\s*Dash", re.I), "200m"),
    (re.compile(r"400m\s*Dash", re.I), "400m"),
    (re.compile(r"800m\s*Run", re.I), "800m"),
    (re.compile(r"1500m\s*Run", re.I), "1500m"),
    (re.compile(r"5000m\s*Run", re.I), "5000m"),
    (re.compile(r"110m\s*Hurdles?", re.I), "110mH"),
    (re.compile(r"110\s*Hurdles?", re.I), "110mH"),
    (re.compile(r"400m\s*Hurdles?", re.I), "400mH"),
    (re.compile(r"High\s*Jump", re.I), "HJ"),
    (re.compile(r"Pole\s*Vault", re.I), "PV"),
    (re.compile(r"Long\s*Jump", re.I), "LJ"),
    (re.compile(r"Triple\s*Jump", re.I), "TJ"),
    (re.compile(r"Shot\s*Put", re.I), "SP"),
    (re.compile(r"Discus", re.I), "DT"),
    (re.compile(r"Hammer", re.I), "HT"),
    (re.compile(r"Javelin", re.I), "JT"),
]


def _norm_event_header(text: str) -> str | None:
    key = re.sub(r"\s+", " ", text.strip().lower().replace("\xa0", " "))
    key = key.replace("–", "-").replace("—", "-")
    return SUPP_HTML_EVENT_MAP.get(key)


def _is_better_mark(event: str, new: float, old: float | None) -> bool:
    if old is None:
        return True
    if event in FIELD_EVENTS:
        return new > old
    return new < old


def _cell_mark_candidates(text: str) -> list[str]:
    """Pull time/mark tokens from a matrix cell (may contain prelim + final)."""
    if not text or text.strip() in {"", "-", "—", "FS", "NH", "DNS", "DNF", "DQ", "Foul"}:
        return []
    text = text.replace("\xa0", " ").replace(",", ".")  # NC typo 10,99
    # drop place suffixes like (7), (pre), (pre) 11.04 (7)
    parts = re.findall(
        r"(\d+:\d{2}\.\d+|\d+\.\d{2}|\d+\.\d+m|\d+\.\d+)",
        text,
        re.I,
    )
    return parts


def scrape_usi_supplementary(coeffs: dict) -> list[dict]:
    """USI outdoor performance list — undated meet names; emit season bests."""
    path = MEN_DOCS / (
        "2025-26 USI Men's Track & Field Performance List - "
        "University of Southern Indiana Athletics.pdf"
    )
    text = extract_pdf_text(path)
    start = text.find("Outdoor Performance List")
    if start < 0:
        return []
    chunk = text[start:]
    best: dict[tuple[str, str], tuple[float, str]] = {}
    current_event: str | None = None
    row_re = re.compile(
        r"^(?P<athlete>[A-Za-z][A-Za-z\.\'\-]+(?:\s+[A-Za-z][A-Za-z\.\'\-]+)+)"
        r"\.+(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)\.+(?P<meet>.+)$"
    )
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low in USI_EVENT_MAP:
            current_event = USI_EVENT_MAP[low]
            continue
        if current_event is None or current_event.startswith("4x"):
            continue
        if "relay" in low:
            continue
        rm = row_re.match(line)
        if not rm:
            continue
        meet = rm.group("meet").strip()
        if re.search(r"\bIndoors?\b", meet, re.I):
            continue
        result_str, value = parse_mark(current_event, rm.group("mark"))
        if value is None:
            continue
        athlete = re.sub(r"\.+$", "", rm.group("athlete")).strip()
        if not athlete:
            continue
        key = (athlete, current_event)
        prev = best.get(key)
        if prev is None or _is_better_mark(current_event, value, prev[0]):
            best[key] = (value, result_str)

    rows = []
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
                college="Southern Indiana",
                division="NCAA D1",
                meet="season best (undated performance list)",
                source=path.name,
                season_year=2026,
                source_role="supplementary_season_pb",
            )
        )
    return rows


def _scrape_html_event_matrix(
    *,
    coeffs: dict,
    path: Path,
    college: str,
    division: str,
    season_year: int,
) -> list[dict]:
    """Sidearm-style HTML: event header rows + athlete × meet mark matrix."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    table = soup.find("table")
    if table is None:
        return []
    best: dict[tuple[str, str], tuple[float, str]] = {}
    current_event: str | None = None
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["th", "td"])]
        if not cells:
            continue
        # Event header: single cell or first cell spanning
        if len(cells) == 1 or (len(cells) >= 1 and all(not c for c in cells[1:])):
            mapped = _norm_event_header(cells[0])
            if mapped:
                current_event = mapped
            continue
        # Meet header row
        if cells[0].lower() in {"name", ""} and any(
            re.search(r"invite|championship|carnival|meet|open|relays|challenge", c, re.I)
            for c in cells[1:]
        ):
            continue
        if current_event is None or current_event.startswith("4x"):
            continue
        athlete = cells[0].strip()
        if not athlete or athlete.lower() == "name":
            continue
        if re.search(r"invite|championship|carnival|meet", athlete, re.I) and not re.search(
            r"[a-z]", athlete.replace(" ", "")
        ):
            continue
        for cell in cells[1:]:
            for token in _cell_mark_candidates(cell):
                result_str, value = parse_mark(current_event, token)
                if value is None:
                    continue
                key = (athlete, current_event)
                prev = best.get(key)
                if prev is None or _is_better_mark(current_event, value, prev[0]):
                    best[key] = (value, result_str)

    rows = []
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
                college=college,
                division=division,
                meet="season best (undated HTML matrix)",
                source=path.name,
                season_year=season_year,
                source_role="supplementary_season_pb",
            )
        )
    return rows


def scrape_north_central_supplementary(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / (
        "2025-26 Men's Outdoor Track & Field Statistics - "
        "North Central College Athletics.html"
    )
    return _scrape_html_event_matrix(
        coeffs=coeffs,
        path=path,
        college="North Central",
        division="NCAA D3",
        season_year=2026,
    )


def scrape_oshkosh_supplementary(coeffs: dict) -> list[dict]:
    path = MEN_DOCS / (
        "2026 Men's Outdoor Track & Field Statistics - "
        "University of Wisconsin-Oshkosh Athletics.html"
    )
    return _scrape_html_event_matrix(
        coeffs=coeffs,
        path=path,
        college="Wisconsin-Oshkosh",
        division="NCAA D3",
        season_year=2026,
    )


def scrape_keiser_supplementary(coeffs: dict) -> list[dict]:
    """Keiser athlete×meet cells list '100m Dash - 10.76 (3rd)' style entries."""
    path = MEN_DOCS / "2026 MOTF Stats - Keiser University Athletics.html"
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    table = soup.find("table")
    if table is None:
        return []
    best: dict[tuple[str, str], tuple[float, str]] = {}
    mark_re = re.compile(
        r"(?P<label>[A-Za-z0-9][A-Za-z0-9 /\-]{2,30}?)\s*-\s*"
        r"(?P<mark>\d+:\d{2}\.\d+|\d+\.\d+m?|\d+\.\d+)",
        re.I,
    )
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True).replace("\xa0", " ") for c in tr.find_all(["th", "td"])]
        if not cells or len(cells) < 2:
            continue
        athlete = cells[0].strip()
        if not athlete or "Track and Field" in athlete:
            continue
        for cell in cells[1:]:
            if not cell:
                continue
            for m in mark_re.finditer(cell):
                label = m.group("label").strip()
                if re.search(r"relay|decathlon|4x", label, re.I):
                    continue
                event = None
                for pat, code in KEISER_EVENT_PATTERNS:
                    if pat.search(label):
                        event = code
                        break
                if event is None:
                    continue
                result_str, value = parse_mark(event, m.group("mark").replace(":", ":"))
                # Keiser typo 55:54 for hurdles — reject absurd hurdle times > 80s for 400H? allow
                if value is None:
                    continue
                # Fix colon-as-decimal typo like 55:54 meaning 55.54 for hurdles under 2 min
                raw_mark = m.group("mark")
                if event in {"110mH", "400mH"} and ":" in raw_mark:
                    mm = re.fullmatch(r"(\d+):(\d{2})", raw_mark)
                    if mm and int(mm.group(1)) < 60:
                        alt = float(f"{mm.group(1)}.{mm.group(2)}")
                        if event == "110mH" and 12 < alt < 25:
                            value = alt
                            result_str = f"{alt:.2f}"
                        if event == "400mH" and 45 < alt < 75:
                            value = alt
                            result_str = f"{alt:.2f}"
                key = (athlete, event)
                prev = best.get(key)
                if prev is None or _is_better_mark(event, value, prev[0]):
                    best[key] = (value, result_str)

    rows = []
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
                college="Keiser",
                division="NAIA",
                meet="season best (undated HTML matrix)",
                source=path.name,
                season_year=2026,
                source_role="supplementary_season_pb",
            )
        )
    return rows


def make_row(
    *,
    coeffs: dict,
    athlete: str,
    event: str,
    result_str: str,
    mark_value: float,
    indoor_outdoor: str,
    competition_date: date | None,
    college: str,
    division: str,
    meet: str,
    source: str,
    wind: str | None = None,
    place: int | None = None,
    season_year: int | None = None,
    source_role: str = "primary_dated",
) -> dict:
    men_pts = wa_points(coeffs, "men", event, mark_value)
    women_pts = wa_points(coeffs, "women", event, mark_value)
    multi_men = multi_score(mark_value, event, "men", coeffs=coeffs)
    multi_women = multi_score(mark_value, event, "women", coeffs=coeffs)
    if season_year is None and competition_date is not None:
        season_year = competition_date.year
    return {
        "Athlete": athlete,
        "College": college,
        "College_Division": division,
        "Event": event,
        "Result": result_str,
        "Result_Value": mark_value,
        "Indoor_Outdoor": indoor_outdoor,
        "Competition_Date": competition_date.isoformat() if competition_date else None,
        "Meet": meet,
        "Place": place,
        "Wind": float(wind) if wind not in (None, "") else None,
        "World_Athletics_Score_Men": men_pts,
        "World_Athletics_Score_Women": women_pts,
        "VDOT_Men": multi_men["vdot"],
        "VDOT_Women": multi_women["vdot"],
        "Purdy_Points_Men": multi_men["purdy"],
        "Purdy_Points_Women": multi_women["purdy"],
        "Mercier_Points_Men": multi_men["mercier"],
        "Mercier_Points_Women": multi_women["mercier"],
        "Source_File": source,
        "Gender": "Men",
        "Season_Year": season_year,
        "Source_Role": source_role,
    }


def event_group_for(event: str) -> str | None:
    for g, events in EVENT_GROUPS.items():
        if event in events:
            return g
    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Source_Role" not in df.columns:
        df["Source_Role"] = "primary_dated"
    df["Competition_Date"] = pd.to_datetime(df["Competition_Date"], errors="coerce")
    df["Event_Group"] = df["Event"].map(event_group_for)

    # Season year: keep pre-set values for supplementary undated rows
    if "Season_Year" not in df.columns:
        df["Season_Year"] = pd.NA
    df["Season_Year"] = df["Season_Year"].fillna(df["Competition_Date"].dt.year)
    df["Season_Year"] = df["Season_Year"].astype("Int64")

    df = df.sort_values(
        ["College", "Athlete", "Season_Year", "Competition_Date", "Event"],
        na_position="last",
    )

    # Chronological markers only for dated rows
    df["Result_Order_In_Season"] = pd.NA
    df["Season_Day_Index"] = pd.NA
    df["Is_First_Result_Of_Season"] = pd.NA
    dated = df["Competition_Date"].notna()
    if dated.any():
        dated_idx = df.index[dated]
        orders = (
            df.loc[dated]
            .groupby(["College", "Athlete", "Season_Year"], dropna=False)
            .cumcount()
            + 1
        )
        df.loc[dated_idx, "Result_Order_In_Season"] = orders.to_numpy()
        season_start = (
            df.loc[dated]
            .groupby(["College", "Athlete", "Season_Year"])["Competition_Date"]
            .transform("min")
        )
        df.loc[dated_idx, "Season_Day_Index"] = (
            df.loc[dated, "Competition_Date"] - season_start
        ).dt.days.to_numpy()
        df.loc[dated_idx, "Is_First_Result_Of_Season"] = (
            df.loc[dated_idx, "Result_Order_In_Season"] == 1
        ).to_numpy()

    # Season PB by event (best = highest WA points for that row's gender tables)
    df["Primary_WA"] = df.apply(
        lambda r: r["World_Athletics_Score_Men"]
        if r.get("Gender", "Men") == "Men"
        else r["World_Athletics_Score_Women"],
        axis=1,
    )
    pb_idx = (
        df.dropna(subset=["Primary_WA"])
        .sort_values("Primary_WA", ascending=False)
        .groupby(["College", "Athlete", "Season_Year", "Event"], as_index=False)
        .head(1)
        .index
    )
    df["Is_Season_Event_PB"] = False
    df.loc[pb_idx, "Is_Season_Event_PB"] = True

    # bal_spec / best_event per athlete-season within each event group
    df["WA_Spread"] = pd.NA
    df["Bal_Spec"] = pd.NA
    df["Best_Event"] = pd.NA
    df["Events_Competed_In_Group"] = pd.NA

    for (college, athlete, season, group), gdf in df.groupby(
        ["College", "Athlete", "Season_Year", "Event_Group"], dropna=True
    ):
        if group is None or pd.isna(group):
            continue
        event_best = gdf.dropna(subset=["Primary_WA"]).groupby("Event")["Primary_WA"].max()
        if len(event_best) == 0:
            continue
        wa_spread = float(event_best.max() - event_best.min()) if len(event_best) >= 2 else 0.0
        bal_spec = "specialized" if wa_spread >= SPREAD_THRESHOLD else "balanced"
        best_event = event_best.idxmax()
        n_events = int(event_best.shape[0])
        idx = gdf.index
        df.loc[idx, "WA_Spread"] = wa_spread
        df.loc[idx, "Bal_Spec"] = bal_spec
        df.loc[idx, "Best_Event"] = best_event
        df.loc[idx, "Events_Competed_In_Group"] = n_events

    df["Best_Is_This_Event"] = df["Event"] == df["Best_Event"]

    def _events_bucket(n):
        if n is pd.NA or (isinstance(n, float) and pd.isna(n)):
            return pd.NA
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            return pd.NA
        if n_int == 2:
            return "2"
        if n_int >= 3:
            return "3+"
        return pd.NA

    df["Events_Competed_Bucket"] = df["Events_Competed_In_Group"].map(_events_bucket)

    # Tolerance helpers for later predicted-vs-actual checks
    def tol_for(row):
        ev = row["Event"]
        if ev in TOLERANCE_SECONDS:
            return TOLERANCE_SECONDS[ev]
        if ev in TOLERANCE_METRES:
            return TOLERANCE_METRES[ev]
        return pd.NA

    df["Tolerance"] = df.apply(tol_for, axis=1)
    df["Tolerance_Unit"] = df["Event"].map(
        lambda e: "seconds" if e in TRACK_EVENTS else ("metres" if e in FIELD_EVENTS else pd.NA)
    )

    # Keep ISO date strings in export (blank for undated supplementary rows)
    df["Competition_Date"] = df["Competition_Date"].dt.strftime("%Y-%m-%d")
    df["Competition_Date"] = df["Competition_Date"].where(df["Competition_Date"].notna(), None)
    return df


def athlete_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per athlete-season-group with bal_spec / best_event."""
    cols = [
        "College",
        "Athlete",
        "Season_Year",
        "Event_Group",
        "Bal_Spec",
        "Best_Event",
        "WA_Spread",
        "Events_Competed_In_Group",
    ]
    sub = df.dropna(subset=["Event_Group"]).copy()
    return (
        sub.sort_values("WA_Spread", ascending=False)
        .groupby(["College", "Athlete", "Season_Year", "Event_Group"], as_index=False)
        .first()[cols]
    )


def run_preprocess() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeffs = load_coefficients()

    primary_scrapers = [
        ("SIUE", scrape_siue),
        ("North Florida", scrape_north_florida),
        ("DePaul", scrape_depaul),
    ]
    supplementary_scrapers = [
        ("USI (suppl.)", scrape_usi_supplementary),
        ("North Central (suppl.)", scrape_north_central_supplementary),
        ("Wisconsin-Oshkosh (suppl.)", scrape_oshkosh_supplementary),
        ("Keiser (suppl.)", scrape_keiser_supplementary),
    ]

    all_rows: list[dict] = []
    counts: dict[str, int] = {}
    for label, fn in primary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"Scraped {label}: {len(part)} dated results")

    for label, fn in supplementary_scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"Scraped {label}: {len(part)} season-best rows (undated)")

    raw = pd.DataFrame(all_rows)
    if raw.empty:
        raise SystemExit("No rows scraped — check parsers.")

    # Deduplicate exact repeats
    before = len(raw)
    raw = raw.drop_duplicates(
        subset=["College", "Athlete", "Event", "Result", "Competition_Date", "Meet"]
    )
    print(f"Deduped {before - len(raw)} duplicate rows; {len(raw)} remain")

    featured = add_features(raw)
    summary = athlete_feature_summary(featured)

    core_cols = [
        "Athlete",
        "College",
        "College_Division",
        "Event",
        "Event_Group",
        "Result",
        "Result_Value",
        "Indoor_Outdoor",
        "Competition_Date",
        "Meet",
        "Place",
        "Wind",
        "World_Athletics_Score_Men",
        "World_Athletics_Score_Women",
        "Season_Year",
        "Result_Order_In_Season",
        "Season_Day_Index",
        "Is_First_Result_Of_Season",
        "Is_Season_Event_PB",
        "WA_Spread",
        "Bal_Spec",
        "Best_Event",
        "Best_Is_This_Event",
        "Events_Competed_In_Group",
        "Events_Competed_Bucket",
        "Tolerance",
        "Tolerance_Unit",
        "Source_File",
        "Source_Role",
        "Gender",
    ]
    featured = featured[core_cols]

    results_path = OUT_DIR / "men_test_dataset_results.csv"
    features_path = OUT_DIR / "men_athlete_season_features.csv"
    suppl_path = OUT_DIR / "men_supplementary_season_pb.csv"
    readme_path = OUT_DIR / "README.txt"
    featured.to_csv(results_path, index=False)
    summary.to_csv(features_path, index=False)
    suppl = featured[featured["Source_Role"] == "supplementary_season_pb"]
    suppl.to_csv(suppl_path, index=False)

    n_athletes = featured.groupby(["College", "Athlete", "Season_Year"]).ngroups
    n_primary = int((featured["Source_Role"] == "primary_dated").sum())
    n_suppl = int((featured["Source_Role"] == "supplementary_season_pb").sum())
    n_with_bal = summary["Bal_Spec"].notna().sum()
    bal_counts = summary["Bal_Spec"].value_counts(dropna=True).to_dict()

    readme = "\n".join(
        [
            "Men's Test Dataset — Time Models",
            "================================",
            "",
            "Primary sources (per-result competition dates; season-PB + chronological):",
            f"  • SIUE outdoor performance list → {counts.get('SIUE', 0)} rows",
            f"  • North Florida 2024 outdoor results → {counts.get('North Florida', 0)} rows",
            f"  • DePaul 2026 outdoor results (TF_2025_Results.pdf, Men's pages only)"
            f" → {counts.get('DePaul', 0)} rows",
            "",
            "Supplementary sources (undated; season-best rows only for season-PB validation):",
            f"  • USI outdoor performance list → {counts.get('USI (suppl.)', 0)} rows",
            f"  • North Central outdoor HTML matrix → {counts.get('North Central (suppl.)', 0)} rows",
            f"  • Wisconsin-Oshkosh outdoor HTML matrix → {counts.get('Wisconsin-Oshkosh (suppl.)', 0)} rows",
            f"  • Keiser outdoor HTML matrix → {counts.get('Keiser (suppl.)', 0)} rows",
            "  • DePaul Women's pages inside TF_2025_Results.pdf (men folder run) — skipped",
            "",
            "Design note:",
            "  Season-PB transfer is the stronger external-validation protocol for these",
            "  club-fit models. Supplementary undated lists expand that protocol. Chronological",
            "  early→later checks use Source_Role=primary_dated rows only.",
            "",
            f"Total result rows: {len(featured)}  (primary_dated={n_primary}, "
            f"supplementary_season_pb={n_suppl})",
            f"Athlete-seasons: {n_athletes}",
            f"Athlete-season-groups with bal_spec: {n_with_bal}  ({bal_counts})",
            "",
            "World Athletics scores:",
            "  Approx. outdoor 2025 quadratic coefficients from",
            "  https://github.com/jchen1/iaaf-scoring-tables (coefficients-2025.json).",
            "",
            "Feature definitions:",
            f"  • bal_spec: balanced if WA_Spread < {SPREAD_THRESHOLD:.0f}; else specialized",
            "  • best_event: event with highest season-best World_Athletics_Score_Men in group",
            "  • Source_Role: primary_dated | supplementary_season_pb",
            "  • Tolerance: starter absolute error band for predicted vs actual",
            "",
            "Outputs:",
            f"  • {results_path.name}",
            f"  • {features_path.name}",
            f"  • {suppl_path.name}",
            "",
        ]
    )
    readme_path.write_text(readme)
    print(readme)
    print(f"Wrote {results_path}")
    print(f"Wrote {features_path}")
    print(f"Wrote {suppl_path}")


def main() -> None:
    run_preprocess()


if __name__ == "__main__":
    main()

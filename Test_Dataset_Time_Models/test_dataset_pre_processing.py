"""Build the men's test dataset for time-model validation.

Scrapes dated competition results from Test_Dataset_Documents/Men,
scores them with World Athletics 2025 outdoor tables, and adds the
feature columns described in this module's design notes:

  • bal_spec / best_event (and related routing flags)
  • chronological markers (result_order_in_season, season_day_index)
  • starter tolerance bands for predicted-vs-actual checks

Only source files with per-result competition dates are included.
HTML season-stat matrices and undated performance lists are skipped.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parent
MEN_DOCS = ROOT / "Test_Dataset_Documents" / "Men"
OUT_DIR = ROOT / "output"
COEFF_PATH = ROOT / "wa_scoring" / "coefficients-2025.json"
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
    with open(COEFF_PATH) as f:
        return json.load(f)


def wa_points(coeffs: dict, gender: str, event: str, mark_value: float) -> int | None:
    key = WA_EVENT_KEYS.get(event)
    if not key or key not in coeffs.get(gender, {}):
        return None
    a, b, c = coeffs[gender][key]
    raw = a * mark_value * mark_value + b * mark_value + c
    if not math.isfinite(raw):
        return None
    pts = int(math.floor(raw))
    return max(pts, 0)


def parse_time_to_seconds(text: str) -> float | None:
    t = text.strip()
    # Fix NF typo style 9.00.49 -> 9:00.49
    if re.fullmatch(r"\d+\.\d{2}\.\d{2}", t):
        parts = t.split(".")
        t = f"{parts[0]}:{parts[1]}.{parts[2]}"
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t)
    m = re.fullmatch(r"(\d+):(\d+):(\d+(?:\.\d+)?)", t)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", t)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


def parse_mark(event: str, raw: str) -> tuple[str, float | None]:
    raw = raw.strip().rstrip("*")
    if raw.upper() in {"ND", "NM", "DNS", "DNF", "DQ", "FS", "FOUL", "NH", "—", "-"}:
        return raw, None
    if event in FIELD_EVENTS:
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*m?", raw, re.I)
        if not m:
            return raw, None
        metres = float(m.group(1))
        # WA jump coeffs expect centimetres for HJ/PV/LJ/TJ in some tables;
        # jchen coefficients were fit on metres for field (validated via SP/HT scale).
        return f"{metres:.2f}m" if "." in m.group(1) else f"{metres}m", metres
    seconds = parse_time_to_seconds(raw)
    if seconds is None:
        return raw, None
    return raw, seconds


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


def make_row(
    *,
    coeffs: dict,
    athlete: str,
    event: str,
    result_str: str,
    mark_value: float,
    indoor_outdoor: str,
    competition_date: date,
    college: str,
    division: str,
    meet: str,
    source: str,
    wind: str | None = None,
    place: int | None = None,
) -> dict:
    men_pts = wa_points(coeffs, "men", event, mark_value)
    women_pts = wa_points(coeffs, "women", event, mark_value)
    return {
        "Athlete": athlete,
        "College": college,
        "College_Division": division,
        "Event": event,
        "Result": result_str,
        "Result_Value": mark_value,
        "Indoor_Outdoor": indoor_outdoor,
        "Competition_Date": competition_date.isoformat(),
        "Meet": meet,
        "Place": place,
        "Wind": float(wind) if wind not in (None, "") else None,
        "World_Athletics_Score_Men": men_pts,
        "World_Athletics_Score_Women": women_pts,
        "Source_File": source,
        "Gender": "Men",
    }


def event_group_for(event: str) -> str | None:
    for g, events in EVENT_GROUPS.items():
        if event in events:
            return g
    return None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Competition_Date"] = pd.to_datetime(df["Competition_Date"])
    df["Event_Group"] = df["Event"].map(event_group_for)

    # Chronological markers within athlete-season (college + calendar year of outdoor season)
    df["Season_Year"] = df["Competition_Date"].dt.year
    df = df.sort_values(["College", "Athlete", "Season_Year", "Competition_Date", "Event"])
    df["Result_Order_In_Season"] = (
        df.groupby(["College", "Athlete", "Season_Year"]).cumcount() + 1
    )
    season_start = df.groupby(["College", "Athlete", "Season_Year"])["Competition_Date"].transform("min")
    df["Season_Day_Index"] = (df["Competition_Date"] - season_start).dt.days
    df["Is_First_Result_Of_Season"] = df["Result_Order_In_Season"] == 1

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
    df["Events_Competed_Bucket"] = df["Events_Competed_In_Group"].map(
        lambda n: "2" if n == 2 else ("3+" if isinstance(n, (int, float)) and n >= 3 else pd.NA)
    )

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

    # Keep ISO date strings in export
    df["Competition_Date"] = df["Competition_Date"].dt.strftime("%Y-%m-%d")
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coeffs = load_coefficients()

    skipped = [
        "2025-26 USI Men's Track & Field Performance List - University of Southern Indiana Athletics.pdf",
        "2025-26 Men's Outdoor Track & Field Statistics - North Central College Athletics.html",
        "2026 Men's Outdoor Track & Field Statistics - University of Wisconsin-Oshkosh Athletics.html",
        "2026 MOTF Stats - Keiser University Athletics.html",
    ]

    scrapers = [
        ("SIUE", scrape_siue),
        ("North Florida", scrape_north_florida),
        ("DePaul", scrape_depaul),
    ]
    all_rows: list[dict] = []
    counts = {}
    for label, fn in scrapers:
        part = fn(coeffs)
        counts[label] = len(part)
        all_rows.extend(part)
        print(f"Scraped {label}: {len(part)} dated results")

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

    # Core schema requested in the design notes (+ athlete keys / features)
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
        "Gender",
    ]
    featured = featured[core_cols]

    results_path = OUT_DIR / "men_test_dataset_results.csv"
    features_path = OUT_DIR / "men_athlete_season_features.csv"
    readme_path = OUT_DIR / "README.txt"
    featured.to_csv(results_path, index=False)
    summary.to_csv(features_path, index=False)

    n_athletes = featured.groupby(["College", "Athlete", "Season_Year"]).ngroups
    n_with_bal = summary["Bal_Spec"].notna().sum()
    bal_counts = summary["Bal_Spec"].value_counts(dropna=True).to_dict()

    readme = "\n".join(
        [
            "Men's Test Dataset — Time Models",
            "================================",
            "",
            "Sources used (have competition dates):",
            f"  • SIUE outdoor performance list → {counts.get('SIUE', 0)} rows",
            f"  • North Florida 2024 outdoor results → {counts.get('North Florida', 0)} rows",
            f"  • DePaul 2026 outdoor results (TF_2025_Results.pdf, Men's pages only)"
            f" → {counts.get('DePaul', 0)} rows",
            "",
            "Sources skipped (no per-result competition dates):",
            *[f"  • {s}" for s in skipped],
            "  • DePaul Women's pages inside TF_2025_Results.pdf (Men folder run)",
            "",
            f"Total result rows: {len(featured)}",
            f"Athlete-seasons: {n_athletes}",
            f"Athlete-season-groups with bal_spec: {n_with_bal}  ({bal_counts})",
            "",
            "World Athletics scores:",
            "  Approx. outdoor 2025 quadratic coefficients from",
            "  https://github.com/jchen1/iaaf-scoring-tables (coefficients-2025.json).",
            "  Men's dataset still records World_Athletics_Score_Women for the same mark",
            "  (useful for cross-checks; primary routing uses Men's scores).",
            "",
            "Feature definitions:",
            f"  • bal_spec: balanced if WA_Spread < {SPREAD_THRESHOLD:.0f}; else specialized",
            "    WA_Spread = max(event-best WA) − min(event-best WA) within Event_Group",
            "  • best_event: event with highest season-best World_Athletics_Score_Men in group",
            "  • Result_Order_In_Season / Season_Day_Index: chronological markers for",
            "    early-season → later-season prediction checks",
            "  • Tolerance: starter absolute error band for predicted vs actual",
            "",
            "Outputs:",
            f"  • {results_path.name}",
            f"  • {features_path.name}",
            "",
        ]
    )
    readme_path.write_text(readme)
    print(readme)
    print(f"Wrote {results_path}")
    print(f"Wrote {features_path}")


if __name__ == "__main__":
    main()

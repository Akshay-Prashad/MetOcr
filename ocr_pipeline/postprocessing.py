import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

DAY_RE = re.compile(r"Day\s+(\d+)[:\s]\s*(.*)", re.IGNORECASE)

FIELD_PATTERNS = {
    "attached_therm": r"(?:Attached\s*Thermometer)\s*[:\s]*([^,]+)",
    "baro_uncorrected": r"(?:Barometer\s*Uncorrected|Uncorrected)\s*[:\s]*(\d+\.?\d*)",
    "baro_corrected": r"(?:Barometer\s*Corrected|Corrected)\s*[:\s]*([^,]+)",
    "dry_bulb": r"(?:Dry\s*[Bb]ulb)\s*[:\s]*(\d+\.?\d*)",
    "wet_bulb": r"(?:Wet\s*[Bb]ulb)\s*[:\s]*(\d+\.?\d*)",
    "wind_dir": r"(?:Wind\s*Direction)[:\s]*([A-Z]+[^,]*)",
    "wind_force": r"(?:Wind\s*Force)[:\s]*(\d+[^,]*)",
    "cloud_amount": r"(?:Cloud\s*Amount)[:\s]*(\d+)",
    "cloud_form": r"(?:Cloud\s*Form)[:\s]*([^,]+)",
    "weather": r"Weather[:\s]*([^,]+)",
    "rain_since_last": r"(?:Rain)\s*[:\s]*(\d+\.?\d*)",
}


def parse_output(text: str) -> list[dict]:
    records = []
    for line in text.strip().splitlines():
        line = line.strip()
        m = DAY_RE.match(line)
        if not m:
            continue
        day = int(m.group(1))
        if day < 1 or day > 31:
            continue
        rest = m.group(2)
        rec = {"day": day}
        for field, pattern in FIELD_PATTERNS.items():
            fm = re.search(pattern, rest, re.IGNORECASE)
            if fm:
                val = fm.group(1).strip().strip(".,-\"")
                if val and val.lower() not in ("", "-", "ins", "null", "none"):
                    rec[field] = val
        records.append(rec)
    return records


def postprocess_markdown(md_pages: list[str]) -> pd.DataFrame:
    all_records = []
    for md in md_pages:
        records = parse_output(md)
        all_records.extend(records)

    if not all_records:
        logger.warning("No records parsed from model output")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    if "day" in df.columns:
        df = df.sort_values("day")
    return df

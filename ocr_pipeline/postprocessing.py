#postprocessing.py
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import re

logger = logging.getLogger(__name__)

VALID_WEATHER_CODES = {"B", "C", "R", "O", "G", "F", "S", "T", "B & C", "B & R", "C & R", "O & R"}

def postprocess_markdown(column_data: List[Dict[str, List[str]]]) -> pd.DataFrame:
    """
    Merges column-by-column extraction results into a single DataFrame.

    Args:
        column_data: List of dicts (one per page).
                     Each dict: {col_name: [31 values]}

    Returns:
        DataFrame with rows as days (1-31) and columns as extracted fields.
    """
    if not column_data:
        return pd.DataFrame()

    page_results = column_data[0]

    # Detect and log column confusion (same value for all days)
    for col_name, values in page_results.items():
        non_empty = [v for v in values if v and str(v).strip() not in ["-", ""]]
        if len(non_empty) > 5 and len(set(str(v) for v in non_empty)) == 1:
            logger.warning(
                f"Column '{col_name}' has same value '{non_empty[0]}' for all days — likely column confusion"
            )

    data = []
    for day in range(1, 32):
        day_row = {"Day": day}

        for col_name, values in page_results.items():
            try:
                val = values[day - 1] if len(values) >= day else ""
            except (IndexError, TypeError):
                val = ""

            val = str(val).strip()
            if val in ["-", "null", "nan", "None", ""]:
                val = None

            val = _clean_value(col_name, val)
            day_row[col_name] = val

        data.append(day_row)

    return pd.DataFrame(data)


def _clean_value(col_name: str, val: Any) -> Optional[Any]:
    if val is None:
        return None
    val = str(val).strip()
    if val in ["-", "null", "nan", "None", ""]:
        return None

    # --- Numeric temperature columns (integers, but allow floats like 42.5) ---
    if col_name in ["attached_thermometer", "dry_bulb"]:
        return _to_int_range(val, 10, 120)
    elif col_name == "wet_bulb":
        # Ground truth has decimals e.g. 42.5 — use float, not int
        num = _to_float(val)
        if num is not None and 10 <= num <= 120:
            return num
        return None

    # --- Wind force: ground truth has ranges like "9 to 10", "4 to 7" ---
    elif col_name == "wind_force":
        if re.search(r'\d+\s+to\s+\d+', val):
            return val  # preserve range string as-is
        # Integer 0 is valid (calm)
        if val == "0":
            return 0
        return _to_int_range(val, 0, 12)

    # --- Cloud amount ---
    elif col_name == "cloud_amount":
        return _to_int_range(val, 0, 9)

    # --- Barometers: widen range to cover real data (28.930 – 31.058 seen in GT) ---
    elif col_name in ["barometer_uncorrected", "barometer_corrected"]:
        numeric = _to_float(val)
        if numeric is not None and 25.0 < numeric < 35.0:
            return numeric
        return None

    # --- Wind direction: compass abbreviations + integer 0 for calm ---
    if col_name == "wind_direction":
        # Integer 0 appears in ground truth for calm/no-wind days
        if val in ("0", "0.0"):
            return 0
        abbreviations = {
            "N": "N", "NNE": "NNE", "NE": "NE", "ENE": "ENE",
            "E": "E", "ESE": "ESE", "SE": "SE", "SSE": "SSE",
            "S": "S", "SSW": "SSW", "SW": "SW", "WSW": "WSW",
            "W": "W", "WNW": "WNW", "NW": "NW", "NNW": "NNW",
        }
        return abbreviations.get(val.upper(), val)

    # --- Weather codes ---
    if col_name == "weather":
        return _clean_weather(val)

    # --- Rain: decimal or "tr" for trace ---
    if col_name == "rain_since_last":
        return _clean_rain(val)

    # --- Cloud form: preserve as-is (freeform: "Ci S.", "A. K.", "S K n.") ---
    if col_name == "cloud_form":
        # Only reject obviously wrong values; preserve original capitalisation
        if val in ["-", ""]:
            return None
        return val

    return val


def _clean_weather(val: str) -> Optional[str]:
    """Normalize weather codes: handle 'B & C', single letters, etc."""
    val_stripped = val.strip()

    # Direct match first (case-sensitive, as GT uses uppercase)
    if val_stripped in VALID_WEATHER_CODES:
        return val_stripped

    val_upper = val_stripped.upper()

    # Case-insensitive match
    for code in VALID_WEATHER_CODES:
        if code.upper() == val_upper:
            return code

    # Partial containment match
    for code in VALID_WEATHER_CODES:
        if code.upper() in val_upper or val_upper in code.upper():
            return code

    # Single-char codes
    if len(val_upper) == 1 and val_upper in "BCROGFST":
        return val_upper

    # Return as-is if non-empty (don't silently drop unknown codes)
    return val_stripped if val_stripped else None


def _clean_rain(val: str) -> Optional[Any]:
    """Handle rain values including 'tr' for trace and decimals."""
    val_lower = val.lower().strip()
    if val_lower in ["tr", "trace", "t"]:
        return "tr"
    # Numeric value (may be integer or decimal)
    match = re.search(r'(\d*\.?\d+)', val)
    if match:
        num_str = match.group(1)
        # Return as float if decimal, int if whole number
        num = float(num_str)
        return int(num) if num == int(num) else num
    return None


def _to_int_range(val: Any, min_val: int, max_val: int) -> Optional[int]:
    if val is None:
        return None
    match = re.search(r'(-?\d+)', str(val))
    if match:
        num = int(match.group(1))
        if min_val <= num <= max_val:
            return num
    return None


def _to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    match = re.search(r'(-?\d+)', str(val))
    return int(match.group(1)) if match else None


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    match = re.search(r'(-?\d*\.?\d+)', str(val))
    return float(match.group(1)) if match else None

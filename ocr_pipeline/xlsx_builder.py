import openpyxl
from openpyxl.styles import Font
import pandas as pd


HEADER_ROWS = [
    [None, None, "At 9 A.M. Local Time", None, None, None, None, None, None,
     None, None, None, None, None, None, None, None, None, None,
     "At Local Time", None, None, None, None, None, None, None, None, None,
     None, None, None, None, None, None, None, None, None, None,
     "Extra Observation if taken", None, None, None, None, None, "Remarks"],
    [None, None, "Barometer", None, None, "Temperature", None, None, "Humidity",
     "Wind", None, "Cloud", None, None, "Weather", None, "Rain", None, None,
     "Barometer", None, None, "Temperature", None, None, "Humidity", None, None,
     None, "Wind", None, "Cloud", None, None, "Weather", None, "Temperature",
     None, "Radiation", None, "Earth Temperature", None, None, "Sunshine",
     "symbol", None],
    [None, "Day of the month", "Attached Thermometer", "Uncorrected",
     "correct-ed and reduced to 32 Fahat mean sea level", "Corrected reading of",
     None, "Dew Point", "Vap. Pressure", "Per Cent.", "Direction", "Force (0-12)",
     "Amount(0-10)", "Form", "Direction of lower stratum whence coming",
     "At time of observation", "Since last Observation", "Entered to preceding day",
     "Estimated duration", "Attached Thermometer", "Uncorrected",
     "Corrected and reduced to 32 Fahat mean sea level", "Corrected reading of",
     None, "Dew Point", "Vap Pressure", "Per Cent.", None, "Day of the month",
     "Direction", "Force(0-12)", "Amount(0-10)", "Form",
     "Direction of lower stratum whence coming", "At time of Observation",
     "Since last Observation", "Corrected reading of", None, "Max in Sun",
     "Min on Grass", None, "At 1 ft. depth", "At 4 ft. depth",
     "Duration of Sunshine", None, None],
    [None, None, None, None, None, "Dry bulb", "Wet bulb", None, None, None,
     None, None, None, None, None, None, None, None, None, None, None, None,
     "Dry bulb", "Wet bulb", None, None, None, None, None, None, None, None,
     None, None, None, None, "Max", "Min", None, None, None, None, None, None,
     None, None],
]

COLUMN_COUNT = 46

# Mapping from JSON field -> (morning_xlsx_col, afternoon_xlsx_col, extras_xlsx_col)
# xlsx columns are 1-indexed
FIELD_MAP = {
    "attached_therm":      (3, None, None),
    "baro_uncorrected":    (4, None, None),
    "baro_corrected":      (5, None, None),
    "dry_bulb":            (6, None, None),
    "wet_bulb":            (7, None, None),
    "dew_point":           (8, None, None),
    "vapour_pressure":     (9, None, None),
    "humidity_pct":        (10, None, None),
    "wind_dir":            (11, None, None),
    "wind_force":          (12, None, None),
    "cloud_amount":        (13, None, None),
    "cloud_form":          (14, None, None),
    "weather":             (16, None, None),
    "rain_since_last":     (17, None, None),
    "max_temp":            (None, None, 37),
    "min_temp":            (None, None, 38),
    "sunshine":            (None, None, 44),
    "remarks":             (None, None, 46),
}


def build_xlsx(
    records: list[dict],
    kv_data: dict,
    raw_md_pages: list[str],
    output_path: str,
):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    _write_headers(ws)

    if records:
        _write_records(ws, records)

    ws_raw = wb.create_sheet("Raw OCR")
    row = 1
    for page_num, md in enumerate(raw_md_pages, 1):
        ws_raw.cell(row, 1, f"--- Page {page_num} ---").font = Font(bold=True)
        ws_raw.cell(row + 1, 1, md)
        row += 2
    ws_raw.column_dimensions["A"].width = 120

    wb.save(output_path)


def _write_headers(ws):
    for r_idx, row_data in enumerate(HEADER_ROWS, 5):
        for c_idx, val in enumerate(row_data, 1):
            if val is not None:
                ws.cell(r_idx, c_idx, val)


FIELD_TO_COL = {
    "attached_therm": 2,
    "baro_uncorrected": 3,
    "baro_corrected": 4,
    "dry_bulb": 5,
    "wet_bulb": 6,
    "wind_dir": 10,
    "wind_force": 11,
    "cloud_amount": 12,
    "cloud_form": 13,
    "weather": 15,
    "rain_since_last": 17,
}


def _write_records(ws, records: list[dict]):
    data_start = 14
    for rec in records:
        day = _safe_int(rec.get("day"))
        if day is None or day < 1 or day > 31:
            continue

        xlsx_row = data_start + day - 1
        xlsx_row_46 = [None] * COLUMN_COUNT
        xlsx_row_46[1] = day

        for field, col_idx in FIELD_TO_COL.items():
            val = rec.get(field)
            if val is not None:
                s = str(val).strip()
                if s and s.lower() != "nan":
                    xlsx_row_46[col_idx] = val

        for c_idx, val in enumerate(xlsx_row_46, 1):
            ws.cell(xlsx_row, c_idx, val)


def _safe_int(v):
    if v is None:
        return None
    try:
        return int(float(str(v).strip()))
    except (ValueError, TypeError):
        return None

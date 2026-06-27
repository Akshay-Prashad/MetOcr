#xlsx_builder.py
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

# Mapping from DataFrame column names to XLSX column positions (1-indexed)
FIELD_TO_COL = {
    "attached_thermometer": 3,   # was 2 — off by 1
    "barometer_uncorrected": 4,  # was 3
    "barometer_corrected":   5,  # was 4
    "dry_bulb":              6,  # was 5
    "wet_bulb":              7,  # was 6
    "wind_direction":        11, # was 10
    "wind_force":            12, # was 11
    "cloud_amount":          13, # was 12
    "cloud_form":            14, # was 13
    "weather":               16, # was 15
    "rain_since_last":       17, # correct
}

def build_xlsx(
    df: pd.DataFrame,
    raw_md_pages: list[str | dict],
    output_path: str,
):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    _write_headers(ws)
    _write_data(ws, df)

    ws_raw = wb.create_sheet("Raw OCR")
    row = 1
    for page_num, md in enumerate(raw_md_pages, 1):
        ws_raw.cell(row, 1, f"--- Page {page_num} ---").font = Font(bold=True)
        ws_raw.cell(row + 1, 1, str(md))
        row += 2
    ws_raw.column_dimensions["A"].width = 120

    wb.save(output_path)


def _write_headers(ws):
    for r_idx, row_data in enumerate(HEADER_ROWS, 5):
        for c_idx, val in enumerate(row_data, 1):
            if val is not None:
                ws.cell(r_idx, c_idx, val)


def _write_data(ws, df: pd.DataFrame):
    data_start = 18  # First data row (day 1) — matches ground truth layout
    
    for _, row in df.iterrows():
        day = row.get("Day")
        if day is None:
            continue
        try:
            day_num = int(float(str(day)))
        except (ValueError, TypeError):
            continue
        if day_num < 1 or day_num > 31:
            continue

        xlsx_row = data_start + day_num - 1
        xlsx_row_46 = [None] * COLUMN_COUNT
        xlsx_row_46[1] = day_num  # Day column

        for field, col_idx in FIELD_TO_COL.items():
            val = row.get(field)
            if val is not None and str(val).strip() not in ["", "nan", "null"]:
                xlsx_row_46[col_idx] = val

        for c_idx, val in enumerate(xlsx_row_46, 1):
            ws.cell(xlsx_row, c_idx, val)

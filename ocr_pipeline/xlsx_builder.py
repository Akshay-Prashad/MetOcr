import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import pandas as pd


def build_xlsx(
    tables: list[pd.DataFrame],
    kv_data: dict,
    raw_md_pages: list[str],
    output_path: str,
):
    wb = openpyxl.Workbook()

    ws_tables = wb.active
    ws_tables.title = "Extracted Tables"
    header_fill = PatternFill("solid", start_color="1F4E79", end_color="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", name="Arial")
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    row = 1
    for i, df in enumerate(tables):
        ws_tables.cell(row, 1, f"Table {i+1}").font = Font(bold=True)
        row += 1
        for col_idx, col_name in enumerate(df.columns, 1):
            cell = ws_tables.cell(row, col_idx, str(col_name))
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")
            cell.border = border
        row += 1
        for _, r in df.iterrows():
            for col_idx, val in enumerate(r, 1):
                cell = ws_tables.cell(row, col_idx, val)
                cell.border = border
                cell.alignment = Alignment(wrap_text=True)
            row += 1
        row += 2

    for col in ws_tables.columns:
        max_len = max((len(str(c.value or "")) for c in col), default=8)
        ws_tables.column_dimensions[get_column_letter(col[0].column)].width = min(
            max_len + 4, 50
        )

    ws_kv = wb.create_sheet("Fields")
    ws_kv["A1"] = "Field"
    ws_kv["B1"] = "Value"
    kv_header_fill = PatternFill("solid", start_color="2E75B6", end_color="2E75B6")
    for cell in [ws_kv["A1"], ws_kv["B1"]]:
        cell.font = Font(bold=True, color="FFFFFF", name="Arial")
        cell.fill = kv_header_fill
    for r, (k, v) in enumerate(kv_data.items(), 2):
        ws_kv.cell(r, 1, k)
        ws_kv.cell(r, 2, v)
    ws_kv.column_dimensions["A"].width = 30
    ws_kv.column_dimensions["B"].width = 50

    ws_raw = wb.create_sheet("Raw OCR")
    row = 1
    for page_num, md in enumerate(raw_md_pages, 1):
        ws_raw.cell(row, 1, f"--- Page {page_num} ---").font = Font(bold=True)
        ws_raw.cell(row + 1, 1, md)
        row += 2
    ws_raw.column_dimensions["A"].width = 120

    wb.save(output_path)

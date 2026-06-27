#verify_extraction.py
#!/usr/bin/env python3
"""
Verification script to compare extracted XLSX output with ground truth.
Usage: python verify_extraction.py <extracted.xlsx> <ground_truth.xlsx>
"""
import sys
from pathlib import Path
import openpyxl


def load_sheet(path: str):
    wb = openpyxl.load_workbook(path)
    return wb.active


def normalise(v) -> str:
    """
    Normalise a cell value for comparison.
    - None / blank → ""
    - Floats that are whole numbers → int string  (30.0 → "30")
    - Other floats → rounded to 3dp string        (30.5230001 → "30.523")
    - Everything else → lowercase stripped string
    """
    if v is None:
        return ""
    s = str(v).strip()
    if s in ("", "-", "None", "nan"):
        return ""
    # Try numeric normalisation
    try:
        f = float(s)
        # Whole number stored as float (e.g. 5.0, 30.0)
        if f == int(f) and "." not in s.rstrip("0"):
            return str(int(f))
        # Round to 3 decimal places to avoid float repr noise
        return f"{f:.3f}".rstrip("0").rstrip(".")
    except ValueError:
        pass
    return s.lower()


def compare_sheets(extracted_path: str, ground_truth_path: str):
    ws_ex = load_sheet(extracted_path)
    ws_gt = load_sheet(ground_truth_path)

    results = []
    total_cells = 0
    correct_cells = 0
    mismatched_cells = 0

    # Both extracted and ground truth use row 18 as data start
    gt_data_start = 18
    ex_data_start = 18

    for day in range(1, 32):
        gt_row = gt_data_start + day - 1
        ex_row = ex_data_start + day - 1

        for col in range(1, 47):
            gt_val = ws_gt.cell(gt_row, col).value

            # Skip cells the ground truth leaves blank
            if gt_val is None:
                continue

            ex_val = ws_ex.cell(ex_row, col).value
            total_cells += 1

            gt_norm = normalise(gt_val)
            ex_norm = normalise(ex_val)

            if gt_norm == ex_norm:
                correct_cells += 1
                status = "OK"
            else:
                mismatched_cells += 1
                status = "MISMATCH"
                results.append({
                    "day":    day,
                    "col":    col,
                    "gt_raw": gt_val,
                    "ex_raw": ex_val,
                    "gt":     gt_norm,
                    "ex":     ex_norm,
                })

    accuracy = correct_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\n=== Verification Report ===")
    print(f"Ground Truth : {ground_truth_path}")
    print(f"Extracted    : {extracted_path}")
    print(f"\nTotal cells  : {total_cells}")
    print(f"Correct      : {correct_cells}")
    print(f"Mismatched   : {mismatched_cells}")
    print(f"Accuracy     : {accuracy:.2f}%")

    if results:
        print(f"\n=== Sample Mismatches (first 20) ===")
        print(f"  {'Day':>3}  {'Col':>3}  {'Ground Truth':<20}  {'Extracted':<20}")
        print(f"  {'-'*3}  {'-'*3}  {'-'*20}  {'-'*20}")
        for r in results[:20]:
            print(
                f"  {r['day']:>3}  {r['col']:>3}  "
                f"{str(r['gt_raw']):<20}  {str(r['ex_raw']):<20}"
                f"  (norm: '{r['gt']}' vs '{r['ex']}')"
            )

    # Per-column breakdown
    print(f"\n=== Per-column accuracy ===")
    col_stats: dict[int, dict] = {}
    for day in range(1, 32):
        gt_row = gt_data_start + day - 1
        ex_row = ex_data_start + day - 1
        for col in range(1, 47):
            gt_val = ws_gt.cell(gt_row, col).value
            if gt_val is None:
                continue
            ex_val = ws_ex.cell(ex_row, col).value
            if col not in col_stats:
                col_stats[col] = {"total": 0, "correct": 0}
            col_stats[col]["total"] += 1
            if normalise(ex_val) == normalise(gt_val):
                col_stats[col]["correct"] += 1

    for col, s in sorted(col_stats.items()):
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        bar = "#" * int(pct / 5)
        print(f"  Col {col:>2}: {pct:5.1f}%  {bar}")

    return {
        "total":      total_cells,
        "correct":    correct_cells,
        "mismatched": mismatched_cells,
        "accuracy":   accuracy,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_extraction.py <extracted.xlsx> <ground_truth.xlsx>")
        sys.exit(1)
    compare_sheets(sys.argv[1], sys.argv[2])

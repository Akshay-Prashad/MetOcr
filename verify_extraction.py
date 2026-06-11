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


def compare_sheets(extracted_path: str, ground_truth_path: str):
    ws_ex = load_sheet(extracted_path)
    ws_gt = load_sheet(ground_truth_path)

    results = []
    total_cells = 0
    correct_cells = 0
    mismatched_cells = 0

    # Ground truth: data starts at row 18 (day 1), extracted: data starts at row 14 (day 1)
    gt_data_start = 18
    ex_data_start = 14

    for day in range(1, 32):
        gt_row = gt_data_start + day - 1
        ex_row = ex_data_start + day - 1

        for col in range(1, 47):
            gt_val = ws_gt.cell(gt_row, col).value
            ex_val = ws_ex.cell(ex_row, col).value

            if gt_val is None:
                continue

            total_cells += 1
            if ex_val == gt_val:
                correct_cells += 1
                status = "OK"
            else:
                mismatched_cells += 1
                status = "MISMATCH"

            if status == "MISMATCH":
                results.append({
                    "row": gt_row,
                    "col": col,
                    "gt": str(gt_val),
                    "ex": str(ex_val),
                    "status": status,
                })

    accuracy = correct_cells / total_cells * 100 if total_cells > 0 else 0

    print(f"\n=== Verification Report ===")
    print(f"Ground Truth: {ground_truth_path}")
    print(f"Extracted: {extracted_path}")
    print(f"\nTotal cells: {total_cells}")
    print(f"Correct: {correct_cells}")
    print(f"Mismatched: {mismatched_cells}")
    print(f"Accuracy: {accuracy:.2f}%")

    if results:
        print(f"\n=== Sample Mismatches ===")
        for r in results[:10]:
            print(f"  Day {r['row']-17}, Col {r['col']}: GT={r['gt']}, EX={r['ex']}")

    return {
        "total": total_cells,
        "correct": correct_cells,
        "mismatched": mismatched_cells,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_extraction.py <extracted.xlsx> <ground_truth.xlsx>")
        sys.exit(1)

    compare_sheets(sys.argv[1], sys.argv[2])
"""
Simple Excel OCR Comparison
Quick script to compare two Excel files and show accuracy
"""

import openpyxl
from difflib import SequenceMatcher


def compare_excel_files(ground_truth_path, ocr_path, show_mismatches=True):
    """
    Compare two Excel files and calculate accuracy.
    
    Args:
        ground_truth_path: Path to ground truth file
        ocr_path: Path to OCR file
        show_mismatches: Whether to print sample mismatches
    """
    
    # Load workbooks
    gt_wb = openpyxl.load_workbook(ground_truth_path)
    ocr_wb = openpyxl.load_workbook(ocr_path)
    
    # Get first sheets
    gt_ws = gt_wb.active
    ocr_ws = ocr_wb.active
    
    # Extract all non-empty cells
    gt_cells = {}
    for row in gt_ws.iter_rows(values_only=False):
        for cell in row:
            if cell.value is not None:
                gt_cells[(cell.row, cell.column)] = str(cell.value)
    
    ocr_cells = {}
    for row in ocr_ws.iter_rows(values_only=False):
        for cell in row:
            if cell.value is not None:
                ocr_cells[(cell.row, cell.column)] = str(cell.value)
    
    # Compare
    exact_matches = 0
    mismatches = []
    
    for pos, gt_val in gt_cells.items():
        if pos in ocr_cells:
            if gt_val == ocr_cells[pos]:
                exact_matches += 1
            else:
                similarity = SequenceMatcher(None, gt_val.lower(), ocr_cells[pos].lower()).ratio()
                mismatches.append({
                    'pos': pos,
                    'gt': gt_val[:80],
                    'ocr': ocr_cells[pos][:80],
                    'sim': round(similarity, 2)
                })
    
    # Calculate accuracy
    accuracy = (exact_matches / len(gt_cells) * 100) if gt_cells else 0
    
    # Print results
    print("\n" + "="*70)
    print(f"{'ACCURACY REPORT':^70}")
    print("="*70)
    print(f"Ground Truth File:  {ground_truth_path}")
    print(f"OCR File:           {ocr_path}")
    print("-"*70)
    print(f"Total cells (GT):            {len(gt_cells)}")
    print(f"Exact matches:               {exact_matches}")
    print(f"Mismatches:                  {len(mismatches)}")
    print(f"Missing in OCR:              {len(gt_cells) - exact_matches - len(mismatches)}")
    print("-"*70)
    print(f"🎯 ACCURACY: {accuracy:.2f}%")
    print("="*70)
    
    if show_mismatches and mismatches:
        print("\nTop 10 Mismatches:")
        print("-"*70)
        for i, m in enumerate(sorted(mismatches, key=lambda x: x['sim'])[:10], 1):
            print(f"{i}. Position {m['pos']}")
            print(f"   GT:   {m['gt']}")
            print(f"   OCR:  {m['ocr']}")
            print(f"   Similarity: {m['sim']}")
            print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python simple_compare.py <ground_truth.xlsx> <ocr.xlsx>")
        sys.exit(1)
    
    compare_excel_files(sys.argv[1], sys.argv[2])

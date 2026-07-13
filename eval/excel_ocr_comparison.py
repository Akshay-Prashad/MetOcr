"""
Excel OCR Accuracy Comparison Script
Compares ground truth Excel file with OCR-extracted Excel file
and calculates various accuracy metrics.
"""

import openpyxl
import pandas as pd
from difflib import SequenceMatcher
from typing import Tuple, Dict, List
import json


class ExcelOCRComparator:
    """Compare two Excel files and calculate OCR accuracy metrics."""
    
    def __init__(self, ground_truth_path: str, ocr_path: str):
        """
        Initialize comparator with two Excel files.
        
        Args:
            ground_truth_path: Path to the ground truth Excel file
            ocr_path: Path to the OCR-extracted Excel file
        """
        self.ground_truth_path = ground_truth_path
        self.ocr_path = ocr_path
        self.gt_wb = openpyxl.load_workbook(ground_truth_path)
        self.ocr_wb = openpyxl.load_workbook(ocr_path)
        self.results = {}
        
    def get_all_cells(self, worksheet) -> List[Tuple[int, int, str]]:
        """
        Extract all non-empty cells from worksheet.
        
        Returns:
            List of tuples (row, col, value)
        """
        cells = []
        for row in worksheet.iter_rows(values_only=False):
            for cell in row:
                if cell.value is not None:
                    cells.append((cell.row, cell.column, str(cell.value)))
        return cells
    
    def string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings (0-1).
        Uses SequenceMatcher for fuzzy matching.
        """
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()
    
    def compare_worksheets(self, sheet_idx: int = 0, similarity_threshold: float = 0.95) -> Dict:
        """
        Compare two worksheets and calculate accuracy.
        
        Args:
            sheet_idx: Index of worksheet to compare (default 0 = first sheet)
            similarity_threshold: Threshold for considering a match (0-1)
        
        Returns:
            Dictionary with detailed comparison results
        """
        gt_ws = self.gt_wb.worksheets[sheet_idx]
        ocr_ws = self.ocr_wb.worksheets[sheet_idx]
        
        gt_cells = self.get_all_cells(gt_ws)
        ocr_cells = self.get_all_cells(ocr_ws)
        
        # Convert to dictionaries for easier lookup
        gt_dict = {(r, c): v for r, c, v in gt_cells}
        ocr_dict = {(r, c): v for r, c, v in ocr_cells}
        
        # Statistics
        total_cells_gt = len(gt_dict)
        total_cells_ocr = len(ocr_dict)
        
        exact_matches = 0
        fuzzy_matches = 0
        mismatches = []
        missing_in_ocr = []
        extra_in_ocr = []
        
        # Compare ground truth cells with OCR cells
        for (row, col), gt_value in gt_dict.items():
            if (row, col) in ocr_dict:
                ocr_value = ocr_dict[(row, col)]
                similarity = self.string_similarity(gt_value, ocr_value)
                
                if gt_value == ocr_value:
                    exact_matches += 1
                elif similarity >= similarity_threshold:
                    fuzzy_matches += 1
                else:
                    mismatches.append({
                        'position': f'({row}, {col})',
                        'ground_truth': gt_value[:100],  # Limit display length
                        'ocr': ocr_value[:100],
                        'similarity': round(similarity, 4)
                    })
            else:
                missing_in_ocr.append({
                    'position': f'({row}, {col})',
                    'value': gt_value[:100]
                })
        
        # Find extra cells in OCR
        for (row, col), ocr_value in ocr_dict.items():
            if (row, col) not in gt_dict:
                extra_in_ocr.append({
                    'position': f'({row}, {col})',
                    'value': ocr_value[:100]
                })
        
        # Calculate accuracy metrics
        exact_accuracy = (exact_matches / total_cells_gt * 100) if total_cells_gt > 0 else 0
        fuzzy_accuracy = ((exact_matches + fuzzy_matches) / total_cells_gt * 100) if total_cells_gt > 0 else 0
        
        return {
            'sheet_name': gt_ws.title,
            'ground_truth_cells': total_cells_gt,
            'ocr_cells': total_cells_ocr,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'mismatches': len(mismatches),
            'missing_in_ocr': len(missing_in_ocr),
            'extra_in_ocr': len(extra_in_ocr),
            'exact_accuracy_percent': round(exact_accuracy, 2),
            'fuzzy_accuracy_percent': round(fuzzy_accuracy, 2),
            'details': {
                'mismatches': mismatches[:20],  # Show first 20
                'missing_in_ocr': missing_in_ocr[:10],
                'extra_in_ocr': extra_in_ocr[:10],
                'note': 'Details truncated to first 20 items. Full results saved to JSON.'
            }
        }
    
    def compare_all_sheets(self, similarity_threshold: float = 0.95) -> Dict:
        """
        Compare all sheets in both workbooks.
        
        Returns:
            Dictionary with results for all sheets
        """
        num_sheets = len(self.gt_wb.worksheets)
        all_results = {
            'ground_truth_file': self.ground_truth_path,
            'ocr_file': self.ocr_path,
            'sheets': []
        }
        
        for idx in range(num_sheets):
            result = self.compare_worksheets(idx, similarity_threshold)
            all_results['sheets'].append(result)
        
        # Calculate overall stats
        total_exact = sum(s['exact_matches'] for s in all_results['sheets'])
        total_cells = sum(s['ground_truth_cells'] for s in all_results['sheets'])
        
        all_results['overall_exact_accuracy'] = round((total_exact / total_cells * 100), 2) if total_cells > 0 else 0
        
        self.results = all_results
        return all_results
    
    def print_summary(self):
        """Print a summary of results to console."""
        if not self.results:
            print("No results. Run compare_all_sheets() first.")
            return
        
        print("\n" + "="*70)
        print("EXCEL OCR ACCURACY COMPARISON REPORT")
        print("="*70)
        print(f"Ground Truth: {self.results['ground_truth_file']}")
        print(f"OCR File:     {self.results['ocr_file']}")
        print("="*70)
        
        for sheet in self.results['sheets']:
            print(f"\nSheet: {sheet['sheet_name']}")
            print(f"  Total cells (GT):        {sheet['ground_truth_cells']}")
            print(f"  Total cells (OCR):       {sheet['ocr_cells']}")
            print(f"  Exact matches:           {sheet['exact_matches']}")
            print(f"  Fuzzy matches:           {sheet['fuzzy_matches']}")
            print(f"  Mismatches:              {sheet['mismatches']}")
            print(f"  Missing in OCR:          {sheet['missing_in_ocr']}")
            print(f"  Extra in OCR:            {sheet['extra_in_ocr']}")
            print(f"  ─────────────────────────────────────")
            print(f"  📊 Exact Accuracy:       {sheet['exact_accuracy_percent']}%")
            print(f"  📊 Fuzzy Accuracy:       {sheet['fuzzy_accuracy_percent']}%")
            
            # Show sample mismatches
            if sheet['details']['mismatches']:
                print(f"\n  Sample Mismatches (first 5):")
                for i, mismatch in enumerate(sheet['details']['mismatches'][:5], 1):
                    print(f"    {i}. Position {mismatch['position']}")
                    print(f"       GT:  {mismatch['ground_truth']}")
                    print(f"       OCR: {mismatch['ocr']}")
                    print(f"       Similarity: {mismatch['similarity']}")
        
        print("\n" + "="*70)
        print(f"OVERALL EXACT ACCURACY: {self.results['overall_exact_accuracy']}%")
        print("="*70 + "\n")
    
    def save_results_json(self, output_path: str):
        """Save detailed results to JSON file."""
        if not self.results:
            print("No results. Run compare_all_sheets() first.")
            return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    
    def save_results_csv(self, output_path: str):
        """Save mismatch details to CSV for review."""
        if not self.results:
            print("No results. Run compare_all_sheets() first.")
            return
        
        import csv
        
        all_mismatches = []
        for sheet in self.results['sheets']:
            for mismatch in sheet['details']['mismatches']:
                all_mismatches.append({
                    'sheet': sheet['sheet_name'],
                    'position': mismatch['position'],
                    'ground_truth': mismatch['ground_truth'],
                    'ocr': mismatch['ocr'],
                    'similarity': mismatch['similarity']
                })
        
        if all_mismatches:
            keys = all_mismatches[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_mismatches)
            print(f"Mismatches saved to: {output_path}")
        else:
            print("No mismatches to save.")


def main():
    """Example usage of the comparator."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python excel_ocr_comparison.py <ground_truth.xlsx> <ocr.xlsx>")
        print("\nExample:")
        print("  python excel_ocr_comparison.py MO_9_2_032_c.xlsx MO-9_1_032_c.xlsx")
        sys.exit(1)
    
    ground_truth = sys.argv[1]
    ocr_file = sys.argv[2]
    
    # Create comparator
    comparator = ExcelOCRComparator(ground_truth, ocr_file)
    
    # Run comparison
    comparator.compare_all_sheets(similarity_threshold=0.95)
    
    # Print summary
    comparator.print_summary()
    
    # Save detailed results
    comparator.save_results_json('ocr_comparison_results.json')
    comparator.save_results_csv('ocr_mismatches.csv')


if __name__ == '__main__':
    main()

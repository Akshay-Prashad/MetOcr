"""
End-to-end pipeline: TIFF -> row-band crops -> Qwen2.5-VL-7B extraction
-> merged JSON -> xlsx writeback.

Run on your 4060:
    python run_pipeline.py /path/to/sheet.tif /path/to/template.xlsx /path/to/out.xlsx

For a batch of sheets in a directory:
    python run_pipeline.py --batch /path/to/tifs_dir /path/to/template.xlsx /path/to/out_dir
"""

import argparse
import json
import os
import time

from crop_utils import prepare_crops
from prompts import SYSTEM_PROMPT, build_user_prompt
from qwen_vlm import run_extraction
from xlsx_writeback import write_rows


def process_sheet(tif_path: str, crops_dir: str, rows_per_band: int = 6):
    """Returns list of row dicts (one per day) extracted from one sheet."""
    crops = prepare_crops(tif_path, crops_dir, rows_per_band=rows_per_band)

    all_rows = []
    for crop_path, first_row, last_row in crops:
        print(f"  extracting rows {first_row:02d}-{last_row:02d} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            user_prompt = build_user_prompt(first_row, last_row)
            rows, raw = run_extraction(crop_path, SYSTEM_PROMPT, user_prompt)
            print(f"ok ({time.time()-t0:.1f}s, {len(rows)} rows)")
            all_rows.extend(rows)
        except Exception as e:
            print(f"FAILED: {e}")
            # Save the raw crop path so it's easy to retry/inspect manually
            all_rows.append({"day": None, "_error": str(e), "_crop": crop_path})

    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif_path", help="Path to source TIFF (or dir if --batch)")
    ap.add_argument("template_xlsx", help="Template xlsx with the target schema/layout")
    ap.add_argument("out_path", help="Output xlsx path (or dir if --batch)")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--rows-per-band", type=int, default=6)
    ap.add_argument("--work-dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "crops"))
    ap.add_argument("--keep-json", action="store_true", help="Save intermediate extracted JSON")
    args = ap.parse_args()

    if args.batch:
        os.makedirs(args.out_path, exist_ok=True)
        tifs = [f for f in os.listdir(args.tif_path) if f.lower().endswith((".tif", ".tiff"))]
        print(f"Found {len(tifs)} TIFFs to process")
        for tif_name in tifs:
            tif_path = os.path.join(args.tif_path, tif_name)
            base = os.path.splitext(tif_name)[0]
            print(f"\n=== {tif_name} ===")
            rows = process_sheet(tif_path, args.work_dir, args.rows_per_band)
            if args.keep_json:
                with open(os.path.join(args.out_path, f"{base}.json"), "w") as f:
                    json.dump(rows, f, indent=2)
            out_xlsx = os.path.join(args.out_path, f"{base}.xlsx")
            write_rows(rows, args.template_xlsx, out_xlsx)
    else:
        print(f"=== {os.path.basename(args.tif_path)} ===")
        rows = process_sheet(args.tif_path, args.work_dir, args.rows_per_band)
        if args.keep_json:
            json_path = os.path.splitext(args.out_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"Saved intermediate JSON: {json_path}")
        write_rows(rows, args.template_xlsx, args.out_path)


if __name__ == "__main__":
    main()

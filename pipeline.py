#!/usr/bin/env python3
import argparse
from pathlib import Path

from ocr_pipeline import (
    load_input,
    preprocess_images,
    get_ocr_images_function,
    postprocess_markdown,
    build_xlsx,
)


def run_pipeline(
    input_path: str,
    output_xlsx: str,
    ocr_type: str = "format",
    tables_only: bool = False,
    ocr_model: str = "qwen_vlm",
    load_in_4bit: bool = True,
):
    raw_images = load_input(input_path)
    print(f"[1/5] Loaded {len(raw_images)} page(s) from {Path(input_path).name}")

    clean_images = preprocess_images(raw_images)
    print("[2/5] Pre-processing complete")

    ocr_func = get_ocr_images_function(ocr_model, load_in_4bit=load_in_4bit)
    md_pages = ocr_func(clean_images, ocr_type=ocr_type)
    print("[3/5] OCR inference complete")

    all_tables, all_kv = postprocess_markdown(md_pages, tables_only=tables_only)
    print(f"[4/5] Post-processing: {len(all_tables)} table(s){'' if tables_only else f', {len(all_kv)} field(s)'}")

    build_xlsx(all_tables, all_kv, md_pages, output_xlsx)
    print(f"[5/5] Saved → {output_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF/TIFF → XLSX OCR pipeline using Qwen2.5-VL or GOT-OCR2_0"
    )
    parser.add_argument("input", help="Path to PDF or TIFF file")
    parser.add_argument("output", help="Output .xlsx path")
    parser.add_argument(
        "--ocr-type",
        default="format",
        choices=["ocr", "format", "fine"],
        help="OCR mode: ocr (raw), format (structured, default), fine (fine-grained)",
    )
    parser.add_argument(
        "--ocr-model",
        default="qwen_vlm",
        choices=["qwen_vlm", "got_ocr"],
        help="OCR model to use: qwen_vlm (default) or got_ocr",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Skip key-value field extraction, only extract tables",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load model in 4-bit quantization (default: True)",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_false",
        dest="load_in_4bit",
        help="Disable 4-bit quantization",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output, args.ocr_type,
                tables_only=args.tables_only, ocr_model=args.ocr_model,
                load_in_4bit=args.load_in_4bit)

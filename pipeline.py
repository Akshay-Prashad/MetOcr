#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from ocr_pipeline import (
    build_xlsx,
    get_ocr_images_function,
    load_input,
    postprocess_markdown,
    preprocess_images,
)


def run_pipeline(
    input_path: str,
    output_xlsx: str,
    ocr_model: str = "qwen_vlm",
    load_in_4bit: bool = True,
):
    raw_images = load_input(input_path)
    print(f"[1/5] Loaded {len(raw_images)} page(s) from {Path(input_path).name}")

    clean_images = preprocess_images(raw_images)
    print("[2/5] Pre-processing complete")

    ocr_func = get_ocr_images_function(ocr_model, load_in_4bit=load_in_4bit)
    md_pages = ocr_func(clean_images)
    print("[3/5] OCR inference complete")

    df = postprocess_markdown(md_pages)
    print(f"[4/5] Post-processing: {len(df)} day(s) extracted")

    build_xlsx(df, md_pages, output_xlsx)
    print(f"[5/5] Saved \u2192 {output_xlsx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF/TIFF \u2192 XLSX OCR pipeline using Qwen2.5-VL"
    )
    parser.add_argument("input", help="Path to PDF or TIFF file")
    parser.add_argument("output", help="Output .xlsx path")
    parser.add_argument(
        "--ocr-model",
        default="qwen_vlm",
        choices=["qwen_vlm"],
        help="OCR model to use: qwen_vlm (default)",
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
    run_pipeline(
        args.input,
        args.output,
        ocr_model=args.ocr_model,
        load_in_4bit=args.load_in_4bit,
    )

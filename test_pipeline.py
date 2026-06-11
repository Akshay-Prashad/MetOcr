#!/usr/bin/env python3
"""Test the OCR pipeline with _c.pdf files from the Files/ directory."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

FILES_DIR = Path(__file__).parent / "Files"
OUTPUT_DIR = Path(__file__).parent / "output"


def get_c_pdfs() -> list[Path]:
    pdfs = sorted(FILES_DIR.glob("*_c.pdf"))
    print(f"Found {len(pdfs)} _c.pdf files")
    return pdfs


def test_preprocessing():
    """Test that preprocessing works on a single PDF."""
    from ocr_pipeline.preprocessing import pdf_to_images, preprocess_image

    pdfs = get_c_pdfs()
    if not pdfs:
        print("SKIP: no _c.pdf files found")
        return

    images = pdf_to_images(str(pdfs[0]), dpi=150)
    assert len(images) > 0, f"No pages extracted from {pdfs[0].name}"
    processed = preprocess_image(images[0])
    assert processed is not None
    print(f"  Preprocessing OK: {pdfs[0].name} → {processed.size}")


def test_postprocessing():
    """Test post-processing with synthetic markdown."""
    from ocr_pipeline.postprocessing import clean_markdown, extract_key_values, extract_tables_from_markdown

    sample_md = """| Day | Temp | Rain |
|-----|------|------|
| 1   | 22   | 0    |
| 2   | 24   | 5    |

Station Name: Reading
Date ......... 2024-01-01
"""
    tables = extract_tables_from_markdown(sample_md)
    assert len(tables) == 1, f"Expected 1 table, got {len(tables)}"
    assert tables[0].shape == (2, 3), f"Expected (2,3), got {tables[0].shape}"

    cleaned = clean_markdown(sample_md)
    kv = extract_key_values(cleaned)
    assert "Station Name" in kv
    assert "Date" in kv
    print(f"  Post-processing OK: {len(tables)} table(s), {len(kv)} field(s)")


def test_xlsx_builder():
    """Test XLSX generation."""
    import pandas as pd
    from ocr_pipeline.xlsx_builder import build_xlsx

    tables = [pd.DataFrame({"A": [1, 2], "B": [3, 4]})]
    kv = {"Station": "Reading", "Date": "2024-01-01"}
    raw = ["| A | B |\n|---|---|\n| 1 | 3 |\n| 2 | 4 |"]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_output.xlsx"
        build_xlsx(tables, kv, raw, str(out))
        assert out.exists(), "XLSX file was not created"
        print(f"  XLSX builder OK → {out}")


def test_full_pipeline_single():
    """Run the full pipeline on the first _c.pdf found."""
    from pipeline import run_pipeline

    pdfs = get_c_pdfs()
    if not pdfs:
        print("SKIP: no _c.pdf files found")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{pdfs[0].stem}.xlsx"

    run_pipeline(str(pdfs[0]), str(out_path), ocr_type="format")
    assert out_path.exists()
    print(f"  Full pipeline OK → {out_path}")


def test_got_ocr_inference():
    """Smoke test GOTOCRInference with mocked model."""
    from ocr_pipeline.ocr_inference import GOTOCRInference

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 0

    mock_model = MagicMock()
    mock_model.chat.return_value = "mocked ocr output"
    mock_model.eval.return_value = mock_model

    with patch("ocr_pipeline.ocr_inference.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("ocr_pipeline.ocr_inference.AutoModel.from_pretrained", return_value=mock_model):
            ocr = GOTOCRInference(load_in_4bit=False)
            img = Image.new("RGB", (10, 10))
            result = ocr.run_ocr(img, ocr_type="format")
            assert result == "mocked ocr output"
            mock_model.chat.assert_called_once()
            print("  GOTOCRInference smoke test OK")


def test_finetune():
    """Test finetune module functions."""
    from ocr_pipeline.finetune import get_training_args
    from transformers import TrainingArguments

    args = get_training_args()
    assert isinstance(args, TrainingArguments)
    assert args.fp16 == torch.cuda.is_available()
    print(f"  get_training_args OK (fp16={args.fp16})")


if __name__ == "__main__":
    print("=== OCR Pipeline Tests ===\n")

    print("[test_preprocessing]")
    test_preprocessing()

    print("\n[test_postprocessing]")
    test_postprocessing()

    print("\n[test_xlsx_builder]")
    test_xlsx_builder()

    print("\n[test_full_pipeline_single]")
    test_full_pipeline_single()

    print("\n[test_got_ocr_inference]")
    test_got_ocr_inference()

    print("\n[test_finetune]")
    test_finetune()

    print("\n=== All tests passed ===")

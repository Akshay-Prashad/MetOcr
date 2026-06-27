#test_pipeline.py
#!/usr/bin/env python3
"""Test the OCR pipeline with _c.tif files from the Files/ directory."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from PIL import Image

FILES_DIR = Path(__file__).parent / "Files"
OUTPUT_DIR = Path(__file__).parent / "output"


def get_c_tifs() -> list[Path]:
    tifs = sorted(FILES_DIR.glob("*_c.tif"))
    print(f"Found {len(tifs)} _c.tif files")
    return tifs


def test_preprocessing():
    """Test that preprocessing works on a single TIFF."""
    from ocr_pipeline.preprocessing import preprocess_image, tiff_to_images

    tifs = get_c_tifs()
    if not tifs:
        print("SKIP: no _c.tif files found")
        return

    images = tiff_to_images(str(tifs[0]))
    assert len(images) > 0, f"No pages extracted from {tifs[0].name}"
    processed = preprocess_image(images[0])
    assert processed is not None
    print(f"  Preprocessing OK: {tifs[0].name} -> {processed.size}")


def test_postprocessing():
    """Test post-processing with synthetic column data."""
    from ocr_pipeline.postprocessing import postprocess_markdown

    column_data = [
        {
            "dry_bulb": ["50", "52", "51", "49", "53", "55", "54", "52", "51", "50",
                         "48", "47", "46", "45", "44", "43", "42", "41", "40", "39",
                         "38", "37", "36", "35", "34", "33", "32", "31", "30", "29", "28"],
            "wet_bulb": ["48", "50", "49", "47", "51", "53", "52", "50", "49", "48",
                         "46", "45", "44", "43", "42", "41", "40", "39", "38", "37",
                         "36", "35", "34", "33", "32", "31", "30", "29", "28", "27", "26"],
            "wind_direction": ["N"] * 31,
            "wind_force": ["3"] * 31,
            "cloud_amount": ["5"] * 31,
            "cloud_form": ["Cu"] * 31,
            "weather": ["B"] * 31,
            "rain_since_last": ["0"] * 31,
            "attached_thermometer": ["50"] * 31,
            "barometer_uncorrected": ["29.8"] * 31,
            "barometer_corrected": ["30.1"] * 31,
        }
    ]

    df = postprocess_markdown(column_data)
    assert len(df) == 31, f"Expected 31 rows, got {len(df)}"
    assert "dry_bulb" in df.columns
    assert df["dry_bulb"].iloc[0] == 50
    print(f"  Post-processing OK: {len(df)} rows, {len(df.columns)} columns")


def test_xlsx_builder():
    """Test XLSX generation."""
    from ocr_pipeline.xlsx_builder import build_xlsx

    df = pd.DataFrame({
        "Day": list(range(1, 32)),
        "dry_bulb": [50] * 31,
        "wet_bulb": [48] * 31,
        "wind_direction": ["N"] * 31,
    })
    raw_md = ["D1: dry_bulb=50, wet_bulb=48"]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_output.xlsx"
        build_xlsx(df, raw_md, str(out))
        assert out.exists(), "XLSX file was not created"
        print(f"  XLSX builder OK -> {out}")


def test_full_pipeline_single():
    """Run the full pipeline on the first _c.tif found."""
    from pipeline import run_pipeline

    tifs = get_c_tifs()
    if not tifs:
        print("SKIP: no _c.tif files found")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{tifs[0].stem}.xlsx"

    run_pipeline(str(tifs[0]), str(out_path))
    assert out_path.exists()
    print(f"  Full pipeline OK -> {out_path}")


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

    print("\n=== All tests passed ===")

# Met Office Register OCR Pipeline

Extracts handwritten meteorological observation tables from scanned register
pages into the existing xlsx schema, using a quantized open VLM sized for
8GB of VRAM.

## What's here

- `schema.py` — column mapping matching `MO_9_2_029_c.xlsx`'s left-page layout
- `crop_utils.py` — splits the two-page spread, crops the table region, slices into row-bands
- `prompts.py` — schema-locked extraction prompt
- `qwen_vlm.py` — Qwen3-VL-8B-Instruct, 4-bit quantized inference wrapper
- `got_ocr.py` — GOT-OCR2.0 (580M), optional cheap cross-check pass
- `xlsx_writeback.py` — merges extracted JSON into the xlsx, flags low-confidence cells
- `run_pipeline.py` — orchestrates the full flow, single sheet or batch

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -U torch --index-url https://download.pytorch.org/whl/cu121
pip install -U transformers accelerate bitsandbytes qwen-vl-utils pillow openpyxl
# optional, for GOT-OCR2.0 cross-check pass:
pip install -U tiktoken verovio
```

First run downloads Qwen3-VL-8B-Instruct (~16GB fp16 on disk, loaded at 4-bit
into VRAM — expect ~5-6GB used on your 8GB card, leaving headroom for image
tokens).

## Run on one sheet

```bash
python run_pipeline.py \
  MO-9_1_029_c.tif \
  MO_9_2_029_c.xlsx \
  out/MO-9_1_029_c_extracted.xlsx \
  --keep-json
```

## Run on a batch of sheets

```bash
python run_pipeline.py --batch \
  /path/to/tifs_dir \
  MO_9_2_029_c.xlsx \
  /path/to/out_dir
```


## Calibration

`crop_utils.py`'s `top_frac`/`bottom_frac` values in `prepare_crops()` were
calibrated against `MO-9_1_029_c.tif`'s specific scan crop. If sheets in your
batch come from a consistent scanning setup (same scanner, same form,
similar crop margins), these should hold across the whole volume. If a
batch was scanned differently, re-check one sample page first:

```bash
python crop_utils.py /path/to/sample.tif /tmp/calibration_check
```

Then view `/tmp/calibration_check/*_rows01-06.png` and adjust the fractions
if headers are cut off or rows are misaligned.



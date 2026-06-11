# Met-OCR

OCR pipeline for Irish Met Office daily weather observation sheets using Qwen2.5-VL.

Extracts structured tabular data from scanned meteorological forms and outputs XLSX matching the ground-truth format (46-column layout with hierarchical headers).

## Pipeline

```
Input (TIFF/PDF) → Preprocess (deskew, denoise) → Qwen2.5-VL OCR → Parse → XLSX
```

1. **Preprocessing** — Deskews and denoises images, converts to RGB (no binarization — preserves detail for VLMs)
2. **OCR** — Qwen2.5-VL (3B or 7B) with a prompt tailored for weather sheet extraction
3. **Postprocessing** — Regex-based parser extracts field values from labeled model output
4. **XLSX Builder** — Writes data into the exact 46-column × 4-header-row format matching the ground truth

## Usage

```bash
python pipeline.py input.tif output.xlsx
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--ocr-model` | `qwen_vlm` | Model to use (only qwen_vlm supported) |
| `--no-4bit` | (4-bit on) | Disable 4-bit quantization |
| `--load-in-4bit` | True | Enable 4-bit quantization |

## File Structure

```
├── pipeline.py                  # CLI entry point
├── ocr_pipeline/
│   ├── qwen_vlm_ocr.py         # Qwen2.5-VL runner (default: 3B, 4-bit)
│   ├── preprocessing.py         # Image loading, deskew, denoise
│   ├── postprocessing.py        # Parse model output → structured records
│   ├── xlsx_builder.py          # XLSX output (ground-truth 46-col format)
│   └── config.py                # Model selection
├── Files/                       # Test images and ground-truth XLSX
├── output/                      # Pipeline output
└── requirements.txt
```

## Model

Default: `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit quantized, ~4GB VRAM).

To use the 7B model, change `DEFAULT_MODEL_ID` in `ocr_pipeline/qwen_vlm_ocr.py`. The 7B model requires more memory (~8GB with 4-bit, or CPU offloading).


## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (7GB+ VRAM for 3B, 24GB+ for 7B)
- See `requirements.txt` for full list
>>>>>>> 49bbb68 (vlm inteference check)

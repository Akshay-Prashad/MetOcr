"""
GOT-OCR2.0 (580M) -- cheap, fast first-pass OCR for raw cell text, run as a
secondary signal alongside Qwen3-VL. Not schema-aware on its own, but its
raw transcription is useful for cross-checking Qwen's structured reads,
especially on numeric columns. Trivial to run on 8GB; treat it as a sanity
check rather than the primary extractor.

Setup:
    pip install -U transformers tiktoken verovio accelerate
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "stepfun-ai/GOT-OCR2_0"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    print(f"Loading {MODEL_ID} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    _model = _model.eval()
    return _model, _tokenizer


def run_raw_ocr(image_path: str) -> str:
    """Returns raw OCR text for a crop -- use as a cross-check signal,
    not as the structured output."""
    model, tokenizer = load_model()
    # ocr_type='format' asks the model to preserve table-like structure
    # where it can detect it; falls back to plain text otherwise.
    result = model.chat(tokenizer, image_path, ocr_type="format")
    return result

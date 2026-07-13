"""
Qwen3-VL-8B-Instruct, 4-bit quantized, sized to run on an 8GB RTX 4060.

Setup (run once):
    pip install -U transformers accelerate bitsandbytes qwen-vl-utils pillow

Notes on 8GB VRAM:
  - Use 4-bit NF4 quantization (bitsandbytes) -- fp16 8B will NOT fit
    alongside image tokens on 8GB.
  - Keep input crops small (row-bands, not full pages) -- Qwen's
    dynamic resolution means image token count scales with pixel count;
    a full 5481x4439 page can alone blow the budget.
  - min_pixels/max_pixels below caps the image tokenizer so a single
    row-band crop (~1700x400px after our crop_utils slicing) stays cheap.
  - If you still OOM: drop rows_per_band to 4 in crop_utils.py, or switch
    to load_in_4bit with bnb_4bit_compute_dtype=torch.float16 instead of
    bfloat16 if your card prefers it.
"""

import json
import re
import torch
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

_model = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print(f"Loading {MODEL_ID} in 4-bit ... (first run will download ~8GB of weights)")
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=_bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    # Cap image resolution fed to the vision tower -- keeps VRAM bounded
    # regardless of input crop size. Tune down further if you OOM.
    _processor = Qwen3VLProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    return _model, _processor


def extract_json_array(text: str):
    """Model sometimes wraps output in prose or code fences -- strip that."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output:\n{text[:500]}")
    return json.loads(match.group(0))


def run_extraction(image_path: str, system_prompt: str, user_prompt: str,
                    max_new_tokens: int = 1500):
    model, processor = load_model()

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic transcription, not creative generation
        )

    trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    return extract_json_array(output_text), output_text

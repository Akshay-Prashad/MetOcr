import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_TOKENS = 512

COLUMN_DESCRIPTIONS = {
    "attached_thermometer": (
        "Whole number temperature reading in Fahrenheit (e.g., 32, 45, 72). "
        "Appears in column 'Attached Thermometer' directly after the day number."
    ),
    "barometer_uncorrected": (
        "Barometric pressure reading (in inches of mercury) before correction. "
        "Format is usually one decimal place (e.g., 30.6)."
    ),
    "barometer_corrected": (
        "Barometric pressure reading corrected to 32F (e.g., 30.523). "
        "Format is typically three decimal places."
    ),
    "dry_bulb": (
        "Dry bulb temperature reading (in F) from standard thermometer. "
        "Whole number between 10 and 100F."
    ),
    "wet_bulb": (
        "Wet bulb temperature reading (in F) from psychrometer. "
        "Must be <= Dry Bulb value for the same day."
    ),
    "wind_direction": (
        "Wind direction abbreviation (e.g., N, NE, ENE, ESE, SE, SSE, S, SSW, SW, "
        "WSW, W, WNW, NW, NNW)."
    ),
    "wind_force": ("Wind force on Beaufort scale 0-12. Single digit."),
    "cloud_amount": ("Sky coverage in oktas 0-9. Single digit."),
    "cloud_form": ("Cloud form abbreviation (e.g., A.S., C.U., S.T.)."),
    "weather": (
        "Weather condition code (e.g., B, C, R, O, G, F, S, T, 'B & C'). "
        "DO NOT default to '-'."
    ),
    "rain_since_last": (
        "Rainfall amount in inches (decimal, e.g., 0.05, 0.34) or 'tr' for trace."
    ),
}

COLUMN_PROMPT = """You are reading a scanned UK Met Office weather observation sheet.
The table has 31 rows (one per day of the month) and many columns.
Your task: Read ONLY the '{col_name}' column.
{description}

Rules:
- Output EXACTLY 31 lines, one per day
- Format: Day N: value
- If a cell is empty or illegible: Day N: -
- Do NOT read any other column
- Do NOT add explanation or notes"""


class QwenVLMRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        load_in_4bit: bool = False,
        adapter_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = DEFAULT_MAX_TOKENS
        vram = _get_vram_gb()
        use_4bit = load_in_4bit or (vram is not None and vram < 8)

        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map=device,
                quantization_config=quant_config,
                low_cpu_mem_usage=True,
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=device,
            )

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, local_files_only=True)
            logger.info(f"Loaded LoRA adapter from {adapter_path}")

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def close(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @torch.inference_mode()
    def run_ocr(self, pil_img: Image.Image, prompt: str) -> str:
        # Resize to fit in VRAM (matches training preprocessing)
        w, h = pil_img.size
        max_size = 768
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            pil_img = pil_img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        torch.cuda.empty_cache()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def extract_column(self, pil_img: Image.Image, column_name: str) -> List[str]:
        description = COLUMN_DESCRIPTIONS.get(column_name, "")
        prompt = COLUMN_PROMPT.format(col_name=column_name, description=description)
        raw_output = self.run_ocr(pil_img, prompt)

        day_values = [""] * 31
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"Day\s+(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            if match:
                day_num = int(match.group(1))
                if 1 <= day_num <= 31:
                    value = match.group(2).strip()
                    day_values[day_num - 1] = (
                        value if value not in ["-", "", "null"] else ""
                    )

        torch.cuda.empty_cache()
        return day_values


def _get_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return None


def ocr_images(
    images: list[Image.Image],
    load_in_4bit: bool = True,
    adapter_path: str = None,
) -> list[Dict[str, List[str]]]:
    with QwenVLMRunner(load_in_4bit=load_in_4bit, adapter_path=adapter_path) as runner:
        results = []
        for img in (pbar := tqdm(images, desc="Page")):
            pbar.set_description("Page")
            column_values = {}
            col_names = list(COLUMN_DESCRIPTIONS.keys())
            random.shuffle(col_names)

            for col_name in (col_pbar := tqdm(col_names, desc=f"  Col", leave=False)):
                col_pbar.set_description(f"  {col_name}")
                try:
                    values = runner.extract_column(img, col_name)
                    column_values[col_name] = values
                except Exception as e:
                    logger.warning(f"Failed column '{col_name}': {e}")
                    column_values[col_name] = [""] * 31

            results.append(column_values)

        return results

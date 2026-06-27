#qwen_vlm_ocr.py
import logging
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

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_IMAGE_SIZE = 1024

COLUMN_DESCRIPTIONS = {
    "attached_thermometer": (
        "Whole number temperature reading in Fahrenheit (e.g., 32, 45, 72). "
        "Column header: 'Attached Thermometer ins.'"
    ),
    "barometer_uncorrected": (
        "Barometric pressure reading (in inches of mercury) before correction. "
        "Format: one decimal place (e.g., 30.6). Column header: 'Barometer Uncorrected'."
    ),
    "barometer_corrected": (
        "Barometric pressure reading corrected to 32F. "
        "Format: three decimal places (e.g., 30.523). "
        "Column header: 'Corrected and reduced to 32 Fahat mean sea level'."
    ),
    "dry_bulb": (
        "Dry bulb temperature reading (in F) from standard thermometer. "
        "Whole number 10-100F. Column header: 'Dry bulb'."
    ),
    "wet_bulb": (
        "Wet bulb temperature reading (in F) from psychrometer. "
        "May be a decimal (e.g., 42.5). Must be <= Dry Bulb for same day. "
        "Column header: 'Wet bulb'."
    ),
    "wind_direction": (
        "Wind direction abbreviation. Valid values: "
        "N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW. "
        "Use 0 for calm (no wind). Column header: 'Direction'."
    ),
    "wind_force": (
        "Wind force on Beaufort scale 0-12. "
        "May be a single digit (e.g., 5) or a range (e.g., 9 to 10, 4 to 7). "
        "Column header: 'Force (0-12)'."
    ),
    "cloud_amount": (
        "Sky coverage in oktas 0-9. Single digit. Column header: 'Amount(0-10)'."
    ),
    "cloud_form": (
        "Cloud form abbreviation. Preserve exactly as written, including dots and spaces "
        "(e.g., A.S., C.U., S.T., Ci S., K.N., S K n., A. K.). "
        "Column header: 'Form'."
    ),
    "weather": (
        "Weather condition code. Valid values: B, C, R, O, G, F, S, T, B & C, B & R, C & R, O & R. "
        "DO NOT default to '-'. Column header: 'At time of observation'."
    ),
    "rain_since_last": (
        "Rainfall amount in inches (decimal, e.g., 0.05, 0.34) or 'tr' for trace. "
        "Many days will be blank/empty. Column header: 'Since last Observation'."
    ),
}

ALL_COLUMNS = list(COLUMN_DESCRIPTIONS.keys())

# Compact aliases used in multi-column prompt (kept for reference / infer_only mode)
COMPACT_FIELD_NAMES = [
    "attached_thermo", "baro_uncorr", "baro_corr", "dry_bulb", "wet_bulb",
    "wind_dir", "wind_force", "cloud_amt", "cloud_form", "weather", "rain",
]
FIELD_NAME_MAP = dict(zip(COMPACT_FIELD_NAMES, ALL_COLUMNS))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Multi-column prompt kept only for --infer_only diagnostics.
# No example values — the model copies them instead of reading the image.
MULTI_COLUMN_PROMPT = """Extract weather data from this meteorological observation sheet.
The table has 31 daily rows. Read every row carefully — each row has DIFFERENT values.

Fields to extract (name = column header):
- attached_thermo  = Attached Thermometer ins.   (integer °F)
- baro_uncorr      = Barometer Uncorrected        (1 decimal, e.g. 30.6)
- baro_corr        = Corrected/reduced to 32F     (3 decimals, e.g. 30.523)
- dry_bulb         = Dry bulb                     (integer °F)
- wet_bulb         = Wet bulb                     (number °F, may be decimal)
- wind_dir         = Direction                    (compass abbrev. e.g. SSW, WNW, or 0 for calm)
- wind_force       = Force (0-12)                 (integer or range e.g. 9 to 10)
- cloud_amt        = Amount (0-10)                (integer 0-9)
- cloud_form       = Form                         (abbreviation e.g. A.S., Ci S., S K n.)
- weather          = At time of observation       (code e.g. B, C, R, B & C, O)
- rain             = Since last Observation       (decimal inches or tr; blank if none)

Output EXACTLY 31 lines, one per day:
D1: attached_thermo=<val>, baro_uncorr=<val>, baro_corr=<val>, dry_bulb=<val>, wet_bulb=<val>, wind_dir=<val>, wind_force=<val>, cloud_amt=<val>, cloud_form=<val>, weather=<val>, rain=<val>
...
D31: ...

Use - for blank/illegible cells. Output ONLY the 31 lines, nothing else."""


# Single-column prompt — no example values to avoid copying.
SINGLE_COLUMN_PROMPT = """You are reading a scanned UK Met Office weather observation sheet.
The table has 31 rows (one per day of the month) and many columns.

Your task: Read ONLY the '{col_name}' column.
{description}

CRITICAL RULES:
- Each day has a DIFFERENT value. Do NOT copy or repeat the same value for multiple days.
- The column header text is NOT a data value — do not output it.
- Read the actual printed number or text for each row individually.
- Output EXACTLY 31 lines, one per day, in order.
- Format: Day N: value
- If a cell is empty or illegible: Day N: -
- Do NOT read any other column.
- Do NOT add explanation, notes, or headers."""


class QwenVLMRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        load_in_4bit: bool = False,
        adapter_path: Optional[str] = None,
        force_cpu: bool = False,
    ):
        self.model_id = model_id
        self.max_new_tokens = DEFAULT_MAX_TOKENS
        self.max_image_size = DEFAULT_MAX_IMAGE_SIZE

        if force_cpu:
            use_4bit = False
            load_kwargs = dict(
                device_map="cpu",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
            )
        else:
            vram = _get_vram_gb()
            use_4bit = load_in_4bit or (vram is not None and vram < 8)
            load_kwargs = dict(
                device_map=device,
                low_cpu_mem_usage=True,
            )
            if use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["torch_dtype"] = torch.float16

        try:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **load_kwargs
            )
        except Exception:
            load_kwargs.pop("attn_implementation", None)
            logger.info("Flash Attention 2 not available, falling back to eager attention")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, **load_kwargs
            )

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, local_files_only=True
            )
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
        w, h = pil_img.size
        if max(w, h) > self.max_image_size:
            ratio = self.max_image_size / max(w, h)
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
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output = output_text[0].strip()
        # Strip markdown code-block markers if model wraps output
        output = re.sub(r'^```\w*\n?', '', output)
        output = re.sub(r'\n?```\s*$', '', output)
        return output

    def extract_column(self, pil_img: Image.Image, column_name: str) -> List[str]:
        """Extract 31 values for a single named column."""
        description = COLUMN_DESCRIPTIONS.get(column_name, "")
        prompt = SINGLE_COLUMN_PROMPT.format(
            col_name=column_name, description=description
        )
        raw_output = self.run_ocr(pil_img, prompt)

        day_values = [""] * 31
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"D(?:ay)?\s*(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            if match:
                day_num = int(match.group(1))
                if 1 <= day_num <= 31:
                    value = match.group(2).strip()
                    day_values[day_num - 1] = (
                        value if value not in ["-", "", "null"] else ""
                    )

        return day_values

    def extract_all_columns_multi(self, pil_img: Image.Image) -> Dict[str, List[str]]:
        """
        Multi-column extraction in a single pass.
        Only used for diagnostics (--infer_only). The default pipeline uses
        extract_column() per column instead.
        """
        raw_output = self.run_ocr(pil_img, MULTI_COLUMN_PROMPT)

        column_values = {col: [""] * 31 for col in ALL_COLUMNS}
        for line in raw_output.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"D\s*(\d+)\s*:\s*(.+)", line, re.IGNORECASE) or \
                    re.match(r"Day\s+(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
            if not match:
                continue
            day_num = int(match.group(1))
            if not (1 <= day_num <= 31):
                continue
            fields_str = match.group(2).strip()
            field_pairs = re.findall(r'(\w[\w_]*)\s*=\s*([^,]+?)(?:\s*,|$)', fields_str)
            for compact_name, value in field_pairs:
                compact_name = compact_name.strip()
                value = value.strip()
                full_name = FIELD_NAME_MAP.get(compact_name, compact_name)
                if full_name in column_values:
                    column_values[full_name][day_num - 1] = (
                        value if value not in ["-", "", "null"] else ""
                    )

        return column_values


def _get_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    except Exception:
        return None


def _has_column_confusion(
    column_values: Dict[str, List[str]], threshold: int = 5
) -> List[str]:
    confused = []
    for col_name, values in column_values.items():
        non_empty = [v for v in values if v and v.strip() not in ["-", ""]]
        if len(non_empty) <= threshold:
            continue
        unique = set(non_empty)
        if len(unique) == 1:
            confused.append(col_name)
        elif len(non_empty) > 10 and len(unique) <= 3:
            most_common = max(unique, key=lambda x: sum(1 for v in non_empty if v == x))
            repeat_count = sum(1 for v in non_empty if v == most_common)
            if repeat_count > len(non_empty) * 0.6:
                confused.append(col_name)
    return confused


def ocr_images(
    images: list[Image.Image],
    load_in_4bit: bool = True,
    adapter_path: str = None,
    model_id: str = DEFAULT_MODEL_ID,
    force_cpu: bool = False,
) -> list[Dict[str, List[str]]]:
    """
    Main entry point for OCR.

    Extracts each column independently (one inference per column) to avoid
    the repetition/confusion failure mode of the single-pass multi-column approach.
    11 inferences per page is slower but far more reliable for the 3B model.
    """
    with QwenVLMRunner(
        model_id=model_id, load_in_4bit=load_in_4bit,
        adapter_path=adapter_path, force_cpu=force_cpu,
    ) as runner:
        results = []
        for img in tqdm(images, desc="Page"):
            column_values: Dict[str, List[str]] = {}

            for col_name in tqdm(ALL_COLUMNS, desc="  Columns", leave=False):
                logger.info(f"  Extracting column: {col_name}")
                try:
                    values = runner.extract_column(img, col_name)

                    # Sanity-check: warn if still confused after single-column prompt
                    non_empty = [v for v in values if v and v.strip() not in ["-", ""]]
                    if len(non_empty) > 5 and len(set(non_empty)) == 1:
                        logger.warning(
                            f"  '{col_name}' still shows single repeated value "
                            f"'{non_empty[0]}' after single-column extraction"
                        )

                    column_values[col_name] = values
                except Exception as ex:
                    logger.warning(f"  Extraction failed for '{col_name}': {ex}")
                    column_values[col_name] = [""] * 31

            results.append(column_values)

        return results

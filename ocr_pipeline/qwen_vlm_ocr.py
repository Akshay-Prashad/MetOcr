import logging
from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_TOKENS = 2048


class QwenVLMRunner:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        load_in_4bit: bool = False,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens

        vram = _get_vram_gb()
        use_4bit = load_in_4bit or (vram is not None and vram < 8)

        if use_4bit:
            from transformers import BitsAndBytesConfig
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
    def run_ocr(self, pil_img: Image.Image) -> str:
        prompt = self._build_rainfall_prompt()

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
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=0.1,
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
        return output_text[0].strip()

    def _build_rainfall_prompt(self) -> str:
        return """List every day's data from this weather observation table. Output exactly:
Day 1: attached_thermometer, barometer_uncorrected, barometer_corrected, dry_bulb, wet_bulb, wind_direction, wind_force, cloud_amount, cloud_form, weather, rain
Day 2: ...
...
Day 31: ...

Use empty for missing values. Only output these 31 lines."""

    def batch_ocr(
        self,
        images: list[Image.Image],
    ) -> list[str]:
        results = []
        for img in tqdm(images, desc="Qwen2.5-VL OCR"):
            try:
                md = self.run_ocr(img)
                results.append(md)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                results.append("")
        return results


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
) -> list[str]:
    with QwenVLMRunner(load_in_4bit=load_in_4bit) as runner:
        return runner.batch_ocr(images)
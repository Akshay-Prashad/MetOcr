import logging
import os
import tempfile
from typing import Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class GOTOCRInference:
    def __init__(
        self,
        model_id: str = "stepfun-ai/GOT-OCR2_0",
        device: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )

        vram = _get_vram_gb()
        use_4bit = load_in_4bit or (vram is not None and vram < 8)

        if use_4bit:
            load_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device,
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
            )
        else:
            load_kwargs = dict(
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device,
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
                torch_dtype=torch.float16,
            )

        self.model = AutoModel.from_pretrained(model_id, **load_kwargs)
        self.model = self.model.eval()

    def close(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @torch.inference_mode()
    def run_ocr(
        self,
        pil_img: Image.Image,
        ocr_type: str = "format",
    ) -> str:
        for attempt in range(2):
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                pil_img.save(tmp.name)
                result = self.model.chat(
                    self.tokenizer,
                    tmp.name,
                    ocr_type=ocr_type,
                    gradio_input=False,
                )
                return result
            except Exception:
                if attempt == 1:
                    raise
                logger.warning("OCR attempt %d failed, retrying...", attempt + 1)
            finally:
                tmp.close()
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def batch_ocr(
        self,
        images: list[Image.Image],
        ocr_type: str = "format",
    ) -> list[str]:
        results = []
        for img in tqdm(images, desc="OCR"):
            md = self.run_ocr(img, ocr_type=ocr_type)
            results.append(md)
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
    ocr_type: str = "format",
) -> list[str]:
    ocr = GOTOCRInference()
    return ocr.batch_ocr(images, ocr_type=ocr_type)

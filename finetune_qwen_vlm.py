"""
Fine-tune Qwen2.5-VL-3B-Instruct with QLoRA on weather sheet column extraction.

Usage:
  python finetune_qwen_vlm.py --data_dir Files/ --output_dir adapters/ --num_epochs 50
  python finetune_qwen_vlm.py --data_dir Files/ --output_dir adapters/ --infer_only
"""

import argparse
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    get_scheduler,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from ocr_pipeline.preprocessing import load_input, preprocess_images
from ocr_pipeline.qwen_vlm_ocr import ALL_COLUMNS, COLUMN_DESCRIPTIONS, SINGLE_COLUMN_PROMPT
from ocr_pipeline.xlsx_builder import FIELD_TO_COL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMNS = list(FIELD_TO_COL.keys())
AUGMENT_TRANSFORM = T.Compose([
    T.RandomAffine(degrees=1.5, translate=(0.01, 0.01), scale=(0.98, 1.02)),
    T.ColorJitter(brightness=0.1, contrast=0.1),
])


def load_ground_truth(xlsx_path: str) -> Dict[str, List[Optional[Any]]]:
    """Load ground truth XLSX and return {col_name: [31 values]}."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    data_start = 18
    gt = {}
    for col_name, col_idx in FIELD_TO_COL.items():
        gt[col_name] = [ws.cell(data_start + day, col_idx).value for day in range(31)]
    return gt


def build_output_text(values: List[Optional[Any]]) -> str:
    lines = []
    for day_num in range(1, 32):
        val = values[day_num - 1]
        if val is None or str(val).strip() in ["", "-"]:
            lines.append(f"D{day_num}: -")
        else:
            lines.append(f"D{day_num}: {val}")
    return "\n".join(lines)


def process_vision_info_safe(messages):
    from qwen_vl_utils import process_vision_info
    return process_vision_info(messages)


def resize_for_vlm(image: Image.Image, max_size: int = 384) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    ratio = max_size / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return image.resize((new_w, new_h), Image.LANCZOS)


def load_all_data(data_dir: str, max_image_size: int = 768) -> List[Dict]:
    data_dir = Path(data_dir)
    tiff_files = sorted(data_dir.glob("*_c.tif"))
    all_pairs = []

    for tiff_path in tqdm(tiff_files, desc="Loading sheets"):
        stem = tiff_path.stem.replace("MO-9_1_", "MO_9_2_")
        xlsx_path = data_dir / f"{stem}.xlsx"
        if not xlsx_path.exists():
            logger.warning(f"No ground truth for {tiff_path.name}, skipping")
            continue
        raw_images = load_input(str(tiff_path))
        clean_images = preprocess_images(raw_images)
        image = resize_for_vlm(clean_images[0], max_size=max_image_size)
        gt = load_ground_truth(str(xlsx_path))
        for col_name in TARGET_COLUMNS:
            description = COLUMN_DESCRIPTIONS.get(col_name, "")
            prompt = SINGLE_COLUMN_PROMPT.format(col_name=col_name, description=description)
            output_text = build_output_text(gt.get(col_name, [""] * 31))
            all_pairs.append({
                "image": image,
                "col_name": col_name,
                "prompt": prompt,
                "output_text": output_text,
            })

    logger.info(f"Total training pairs: {len(all_pairs)}")
    return all_pairs


class WeatherDataset(Dataset):
    def __init__(self, data_pairs: List[Dict], processor: AutoProcessor, augment: bool = False):
        self.data = data_pairs
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        prompt = item["prompt"]
        output_text = item["output_text"]

        if self.augment:
            image = AUGMENT_TRANSFORM(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info_safe(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        prompt_len = inputs["input_ids"].shape[1]
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs.get("image_grid_thw")

        output_ids = self.processor.tokenizer(
            output_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

        full_input_ids = torch.cat([inputs["input_ids"][0], output_ids])
        full_attention = torch.ones_like(full_input_ids)

        labels = torch.full_like(full_input_ids, -100)
        labels[prompt_len:] = output_ids

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)

    pad_id = 151643  # Qwen2.5-VL pad token ID
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def setup_model(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    adapter_path: Optional[str] = None,
    max_gpu_mem: str = "5GB",
):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    load_kwargs = dict(
        device_map="cpu",
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, **load_kwargs
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded adapter from {adapter_path}")
    else:
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


@torch.inference_mode()
def evaluate(model, val_loader, processor, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        total_loss += outputs.loss.item()
        num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0


def parse_extracted(text: str) -> Dict[int, str]:
    """Parse 'Day N: value' format from model output."""
    result = {}
    for line in text.strip().splitlines():
        m = re.match(r"Day\s+(\d+)\s*:\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            result[int(m.group(1))] = m.group(2).strip()
    return result


def train():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for weather sheet OCR")
    parser.add_argument("--data_dir", default="Files/")
    parser.add_argument("--output_dir", default="adapters_test/")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Load existing adapter to continue training")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (1 due to VRAM)")
    parser.add_argument("--grad_accum_steps", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_only", action="store_true", help="Skip training, just test inference")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    data_pairs = load_all_data(args.data_dir)
    if len(data_pairs) == 0:
        logger.error("No training data found!")
        return

    random.shuffle(data_pairs)
    split_idx = max(1, int(len(data_pairs) * 0.8))
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]
    logger.info(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    model = setup_model(args.model_id, args.adapter_path)
    processor = AutoProcessor.from_pretrained(args.model_id)

    train_ds = WeatherDataset(train_pairs, processor, augment=(not args.infer_only))
    val_ds = WeatherDataset(val_pairs, processor, augment=False) if val_pairs else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0) if val_ds else None

    if args.infer_only:
        logger.info("Inference-only mode. Testing with single-column prompt...")
        item = data_pairs[0]
        image = item["image"]
        prompt = item["prompt"]

        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info_safe(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(device)

        model.eval()
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        generated_trimmed = generated[0][inputs["input_ids"].shape[1]:]
        output = processor.decode(generated_trimmed, skip_special_tokens=True)
        logger.info(f"Column: {item['col_name']}\nOutput:\n{output}")
        return

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = math.ceil(len(train_loader) * args.num_epochs / args.grad_accum_steps)
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    global_step = 0
    last_val_step = -1
    best_val_loss = float("inf")
    logger.info(f"Starting training for {args.num_epochs} epochs, {total_steps} steps, warmup={warmup_steps}")

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}", "lr": f"{current_lr:.2e}"})

            if val_loader and global_step > 0 and global_step % 10 == 0 and global_step != last_val_step:
                last_val_step = global_step
                val_loss = evaluate(model, val_loader, processor, device)
                logger.info(f"Step {global_step}: val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    logger.info(f"Checkpoint saved to {args.output_dir}")
                model.train()

        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg_loss={avg_epoch_loss:.4f}")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"Final adapter saved to {args.output_dir}")


if __name__ == "__main__":
    train()

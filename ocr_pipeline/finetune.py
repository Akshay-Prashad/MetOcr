from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def get_lora_model(
    model_id: str = "stepfun-ai/GOT-OCR2_0",
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.05,
):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def get_training_args(output_dir: str = "./got-ocr-finetuned") -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
    )


def run_training(
    model,
    tokenizer,
    dataset,
    data_collator=None,
    output_dir: str = "./got-ocr-finetuned",
) -> Trainer:
    if data_collator is None:
        data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
    args = get_training_args(output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    return trainer

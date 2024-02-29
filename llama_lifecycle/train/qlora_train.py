import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from datasets import load_from_disk

MODEL_PATH = "../../models/llama-2-7b-chat-hf/"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# Base model (in 8 or 4 bit for QLoRa
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    quantization_config=quantization_config,
    device_map="cuda:0",
    max_memory="12000MB",
)

# Load tokenizer and set new token to attend EOS tokens
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=False,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# Load dataset
train_dataset = load_from_disk("../../datasets/oasst1")
train_dataset = train_dataset.map(
    lambda samples: tokenizer(
        samples["message_tree_text"],
        padding=True,
        truncation=True,
    ),
    batched=True,
)

# Peft config
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


model = prepare_model_for_kbit_training(model)

# PeftModel with Lora adapters
model = get_peft_model(model, peft_config=lora_config)

# train
# HF trainer / HF accelerate
training_arguments = TrainingArguments(
    output_dir="../../models/llama-2-7b-chat-sft/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    max_steps=10,  # 1000
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    gradient_checkpointing=True,
    push_to_hub=False,
)


def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


# Free memory
torch.cuda.empty_cache()

# SFT
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    dataset_text_field="message_tree_text",
    tokenizer=tokenizer,
    max_seq_length=1024,
    callbacks=[PeftSavingCallback()],
)

trainer.train()
model.save_pretrained("../../models/llama-2-7b-chat-sft/qlora_adapter")

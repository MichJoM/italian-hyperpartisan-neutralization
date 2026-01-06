import unsloth
from unsloth import FastModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

os.environ["TRITON_DISABLE"] = "1"  # Disable Triton kernels
os.environ["TORCH_INDUTOR_DISABLE"] = "1"  # Disable inductor backend
os.environ["TORCHDYNAMO_INLINE_INBUILT"] = "1"  # Optional: helps stability
# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "unsloth/gemma-3-4b-it"
TRAIN_FILE = "data/sft_train_original.jsonl"
DEV_FILE = "data/sft_dev_original.jsonl"
OUTPUT_DIR = Path("outputs/gemma3-neutral-rewriter")
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
R = 8
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# -----------------------------
# Load model (we only need the model, not the processor)
# -----------------------------
model, _ = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# Load the **real tokenizer** separately (critical for Gemma-3 multimodal)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Apply the Gemma-3 chat template
tokenizer = get_chat_template(
    tokenizer=tokenizer,  # ← Correct: pass the tokenizer
    chat_template="gemma-3",  # ← Correct: specify the template
)

# Apply LoRA
model = FastModel.get_peft_model(
    model,
    r=R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=TARGET_MODULES,
    bias="none",
    random_state=42,
    finetune_vision_layers=False,  # Turn off for just text!
    finetune_language_layers=True,  # Should leave on!
    finetune_attention_modules=True,  # Attention good for GRPO
    finetune_mlp_modules=True,  # SHould leave on always!
)


# -----------------------------
# Load and format dataset
# -----------------------------
raw_datasets = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "validation": DEV_FILE},
)

INSTRUCTION = "Riscrivi il seguente paragrafo in modo neutrale, rimuovendo ogni tono hyperpartisan, mantenendo i fatti e il significato originale."


def formatting_prompts_func(examples):
    texts = []
    for i in range(len(examples["output"])):
        instruction = (
            examples.get("instruction", [INSTRUCTION])[i]
            if examples.get("instruction")
            else INSTRUCTION
        )
        input_text = examples.get("input", [""])[i] if examples.get("input") else ""
        output = examples["output"][i]

        sys_instr = (instruction or INSTRUCTION).strip()
        user_content = (input_text or "").strip()
        assistant_content = output.strip()

        messages = [{"role": "system", "content": sys_instr}]
        if user_content:
            messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if text.startswith("<bos>"):
                text = text[5:]
            texts.append(text if text.strip() else " ")
        except Exception as e:
            print(f"Error in row {i}: {e}")
            texts.append(" ")
    return {"text": texts}


formatted = raw_datasets.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

formatted["train"] = formatted["train"].filter(lambda x: len(x["text"].strip()) > 0)
formatted["validation"] = formatted["validation"].filter(
    lambda x: len(x["text"].strip()) > 0
)

print(f"Final train size: {len(formatted['train'])}")
print(f"Final validation size: {len(formatted['validation'])}")

val_original = raw_datasets["validation"]

# -----------------------------
# Trainer (no max_seq_length, packing=False)
# -----------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=formatted["train"],
    eval_dataset=formatted["validation"],
    dataset_text_field="text",
    args=SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        bf16=False,
        packing=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataset_num_proc=1,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",
    response_part="<start_of_turn>model\n",
)

# -----------------------------
# Train
# -----------------------------
print("Starting training...")
trainer_stats = trainer.train()

# -----------------------------
# Save
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# -----------------------------
# Generate predictions
# -----------------------------
print("Generating predictions on validation set...")


def make_inference_prompt(row):
    sys_instr = (row.get("instruction") or INSTRUCTION).strip()
    user_content = row["input"].strip() if row.get("input") else ""

    messages = [{"role": "system", "content": sys_instr}]
    if user_content:
        messages.append({"role": "user", "content": user_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


generated_texts = []
batch_size = 4

for i in tqdm(range(0, len(val_original), batch_size)):
    batch = [val_original[j] for j in range(i, min(i + batch_size, len(val_original)))]
    prompts = [make_inference_prompt(row) for row in batch]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )
    generated_texts.extend([t.strip() for t in gen])

# Save CSV
df = pd.DataFrame(
    {
        "input": [r["input"] for r in val_original],
        "output": val_original["output"],
        "generated": generated_texts,
    }
)
df.to_csv(PREDICTIONS_CSV, index=False)
print(f"Predictions saved to {PREDICTIONS_CSV}")

print(
    f"Training completed in {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes"
)
print(f"Peak VRAM: ~{torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

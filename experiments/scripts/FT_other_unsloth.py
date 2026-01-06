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
import argparse

os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_INDUTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_INLINE_INBUILT"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Add this line

# -----------------------------
# Model configurations with chat templates
# -----------------------------
MODEL_CONFIGS = {
    "gemma-3-4b": {
        "model_name": "unsloth/gemma-3-4b-it",
        "chat_template": "gemma-3",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        "max_seq_length": 1024,
    },
    "gemma-2-2b": {
        "model_name": "unsloth/gemma-2-2b-it",
        "chat_template": "gemma",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        "max_seq_length": 8192,
    },
    "llama-3.2-3b": {
        "model_name": "unsloth/Llama-3.2-3B-Instruct",
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_seq_length": 2048,
    },
    "llama-3.2-1b": {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_seq_length": 2048,
    },
    "llama-3.1-8b": {
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "chat_template": "llama-3.1",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "max_seq_length": 2048,
    },
    "qwen-2.5-3b": {
        "model_name": "unsloth/Qwen2.5-3B-Instruct",
        "chat_template": "qwen-2.5",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "max_seq_length": 4096,
    },
    "phi-3.5-mini": {
        "model_name": "unsloth/Phi-3.5-mini-instruct",
        "chat_template": "phi-3.5",
        "instruction_part": "<|user|>\n",
        "response_part": "<|assistant|>\n",
        "max_seq_length": 4096,
    },
    "mistral-7b": {
        "model_name": "unsloth/mistral-7b-instruct-v0.3",
        "chat_template": "mistral",
        "instruction_part": "[INST]",
        "response_part": "[/INST]",
        "max_seq_length": 8192,
    },
}

# -----------------------------
# Configuration
# -----------------------------
INSTRUCTION = "Riscrivi il seguente paragrafo in modo neutrale, rimuovendo ogni tono hyperpartisan, mantenendo i fatti e il significato originale."


def get_model_and_tokenizer(model_key, max_seq_override=None):
    """Load model and tokenizer with appropriate chat template."""
    config = MODEL_CONFIGS[model_key]
    model_name = config["model_name"]
    max_seq_length = max_seq_override or config["max_seq_length"]

    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")

    # Load model
    model, _ = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Load tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer=tokenizer,
        chat_template=config["chat_template"],
    )

    return model, tokenizer, config


def apply_lora(model, r=8, lora_alpha=16, model_family="gemma"):
    """Apply LoRA with model-specific settings."""
    # Gemma-3 has vision components, others don't
    finetune_vision = model_family.startswith("gemma-3")

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    return FastModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        random_state=42,
        finetune_vision_layers=finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )


def formatting_prompts_func(examples, tokenizer):
    """Format examples using model-specific chat template."""
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
            # Remove leading <bos> if present (Gemma specific)
            if text.startswith("<bos>"):
                text = text[5:]
            texts.append(text if text.strip() else " ")
        except Exception as e:
            print(f"Error in row {i}: {e}")
            texts.append(" ")
    return {"text": texts}


def make_inference_prompt(row, tokenizer):
    """Create inference prompt using chat template."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-4b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to fine-tune",
    )
    parser.add_argument(
        "--train_file", type=str, default="data/sft_train_original.jsonl"
    )
    parser.add_argument("--dev_file", type=str, default="data/sft_dev_original.jsonl")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model}-neutral-rewriter"
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
    MODEL_PREDICTIONS_CSV = OUTPUT_DIR / f"{args.model}_predictions.csv"

    # Load model and tokenizer
    model, tokenizer, config = get_model_and_tokenizer(
        args.model, max_seq_override=args.max_seq_length
    )
    MAX_SEQ_LENGTH = args.max_seq_length or config["max_seq_length"]

    # Apply LoRA
    print("Applying LoRA...")
    model = apply_lora(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, model_family=args.model
    )

    # Load and format dataset
    print("Loading dataset...")
    raw_datasets = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.dev_file},
    )

    formatted = raw_datasets.map(
        lambda x: formatting_prompts_func(x, tokenizer),
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

    # Training
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            fp16=False,
            bf16=True,
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

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part=config["instruction_part"],
        response_part=config["response_part"],
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    # Save
    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Generate predictions
    print("Generating predictions on validation set...")
    generated_texts = []
    batch_size = 4

    for i in tqdm(range(0, len(val_original), batch_size)):
        batch = [
            val_original[j] for j in range(i, min(i + batch_size, len(val_original)))
        ]
        prompts = [make_inference_prompt(row, tokenizer) for row in batch]

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
    df.to_csv(MODEL_PREDICTIONS_CSV, index=False)
    print("Predictions saved to %s and %s" % (PREDICTIONS_CSV, MODEL_PREDICTIONS_CSV))

    print(
        f"Training completed in {trainer_stats.metrics['train_runtime'] / 60:.2f} minutes"
    )
    print(f"Peak VRAM: ~{torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Memory-optimized single-GPU fine-tuning for V100 with 4-bit quantization."""

import argparse
import os
import gc
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

# Constants
INSTRUCTION = "Riscrivi il seguente paragrafo in modo neutrale, rimuovendo ogni tono hyperpartisan, mantenendo i fatti e il significato originale."
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def formatting_prompts_func(examples):
    """Format examples into Llama-3 style prompts."""
    outputs = examples.get("output", [])
    num_rows = len(outputs)
    inputs = examples.get("input") or [""] * num_rows
    instructions = examples.get("instruction") or [INSTRUCTION] * num_rows

    texts = []
    for instr, inp, out in zip(instructions, inputs, outputs):
        user_msg = (instr or "").strip()
        if not user_msg:
            user_msg = f"{INSTRUCTION}\n\nParagrafo: {(inp or '').strip()}"

        text = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{out}<|eot_id|>"
        )
        texts.append(text)
    return {"text": texts}


def extract_assistant_response(text: str) -> str:
    """Extract assistant response from formatted text."""
    if "<|start_header_id|>assistant<|end_header_id|>" in text:
        return (
            text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            .split("<|eot_id|>")[0]
            .strip()
        )
    return text.strip()


def compute_metrics_lightweight(eval_preds, tokenizer) -> Dict[str, float]:
    """Lightweight metrics computation using CPU-based evaluation."""
    from evaluate import load as load_metric

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Extract assistant responses
    decoded_preds = [extract_assistant_response(p) for p in decoded_preds]
    decoded_labels = [extract_assistant_response(l) for l in decoded_labels]

    # Compute lightweight metrics only
    try:
        # Load metrics on-demand
        rouge = load_metric("rouge")

        rouge_results = rouge.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        results = {
            "rougeL": rouge_results["rougeL"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
        }

        # Add generation length
        pred_lens = [len(p.split()) for p in decoded_preds]
        results["avg_gen_len"] = np.mean(pred_lens)

        # Clean up
        del rouge
        gc.collect()

        return {
            k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()
        }

    except Exception as e:
        print(f"Metrics computation failed: {e}")
        return {"rougeL": 0.0}


def setup_model_and_tokenizer(args):
    """Initialize model and tokenizer with optimized settings."""

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization config - optimized for V100
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,  # Disable for V100 stability
        bnb_4bit_compute_dtype=torch.float16,  # V100 doesn't support bf16 well
        bnb_4bit_quant_type="nf4",
    )

    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1  # Disable tensor parallelism artifacts

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    return model, tokenizer


def setup_lora(model, args):
    """Configure and apply LoRA to the model."""

    # Determine target modules based on model architecture
    if "gemma" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "llama" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return peft_config


def main():
    parser = argparse.ArgumentParser(description="Memory-optimized LLM fine-tuning")

    # Model and data arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training JSONL file"
    )
    parser.add_argument(
        "--dev_file", type=str, required=True, help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/fine-tuned-model",
        help="Directory to save model checkpoints",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Per-device batch size"
    )
    parser.add_argument(
        "--grad_accum", type=int, default=32, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # System arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cpu_offload", action="store_true", help="Enable CPU offloading for optimizer"
    )
    parser.add_argument(
        "--no_eval", action="store_true", help="Disable evaluation during training"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print("=" * 80)
    print("Configuration:")
    print(f"  Model: {args.model_name}")
    print(f"  Batch size: {args.batch_size} (accum: {args.grad_accum})")
    print(f"  Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Epochs: {args.epochs}")
    print(f"  CPU offload: {args.cpu_offload}")
    print("=" * 80)

    # 1. Load and format dataset
    print("\nLoading dataset...")
    dataset = load_dataset(
        "json", data_files={"train": args.train_file, "validation": args.dev_file}
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Formatting prompts",
    )

    print(f"  Training samples: {len(dataset['train'])}")
    print(f"  Validation samples: {len(dataset['validation'])}")

    # 2. Setup model and tokenizer
    print("\nInitializing model...")
    model, tokenizer = setup_model_and_tokenizer(args)

    # 3. Setup LoRA
    print("Configuring LoRA...")
    peft_config = setup_lora(model, args)

    # 4. Training configuration
    print("Configuring training...")
    sft_config = SFTConfig(
        # Output and logging
        output_dir=args.output_dir,
        run_name=f"{args.model_name.split('/')[-1]}-finetuned",
        # Dataset configuration
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,  # Disable packing to reduce memory
        # Training parameters
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Optimization
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit" if args.cpu_offload else "adamw_8bit",
        # Precision
        fp16=True,
        bf16=False,  # V100 doesn't support bf16 well
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        # Evaluation and saving
        eval_strategy="epoch" if not args.no_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=not args.no_eval,
        metric_for_best_model="rougeL" if not args.no_eval else None,
        greater_is_better=True,
        # Logging
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        # DataLoader settings
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        # Other
        seed=args.seed,
    )

    # 5. Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if not args.no_eval else None,
        processing_class=tokenizer,
        peft_config=peft_config,
        compute_metrics=lambda p: compute_metrics_lightweight(p, tokenizer)
        if not args.no_eval
        else None,
    )

    # 6. Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n" + "=" * 80)
            print("ERROR: Out of memory!")
            print("Try these solutions:")
            print("  1. Reduce --batch_size to 1 (if not already)")
            print("  2. Reduce --max_seq_len (e.g., to 384 or 256)")
            print("  3. Add --no_eval flag to disable evaluation")
            print("  4. Use a smaller model (e.g., gemma-2-2b-it)")
            print("=" * 80)
        raise

    # 7. Save final model
    print("\nSaving model...")
    output_path = args.output_dir
    trainer.model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to: {output_path}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

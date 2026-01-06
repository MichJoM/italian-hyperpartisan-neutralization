#!/usr/bin/env python3
"""
sft_it5.py - Fine-tuning IT5 (gsarti/it5-base) for Italian text neutralization

SPECIFIC FIXES FOR IT5:
1. Use AutoTokenizer instead of T5Tokenizer (IT5 uses T5TokenizerFast)
2. IT5 is pretrained on Italian - no task prefix needed (unlike FLAN-T5)
3. Use text_target parameter for label tokenization (works with T5TokenizerFast)
4. IT5-base has 220M parameters - fits on L4 GPU with LoRA

IT5 DIFFERENCES FROM FLAN-T5:
- IT5 is pretrained on Italian from scratch (not translated)
- IT5 doesn't require task prefixes (not instruction-tuned)
- IT5 uses T5TokenizerFast which works better with AutoTokenizer
"""

from __future__ import annotations

import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer.*")
warnings.filterwarnings("ignore", message=".*as_target_tokenizer.*")
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")


# =============================================================================
# DEBUG CALLBACK
# =============================================================================


class DebugLossCallback(TrainerCallback):
    """Callback to debug loss values during training."""

    def on_step_end(self, args, state, control, **kwargs):
        if DEBUG and state.log_history:
            last_log = state.log_history[-1] if state.log_history else {}
            logger.debug(f"Step {state.global_step}: log_history={last_log}")


class FixedSeq2SeqTrainer(Seq2SeqTrainer):
    """Trainer that properly computes and tracks loss.

    The standard Seq2SeqTrainer sometimes shows loss=0.0.
    This subclass explicitly implements compute_loss to ensure proper loss tracking.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with explicit tracking."""
        if DEBUG:
            logger.debug(f"compute_loss input keys: {list(inputs.keys())}")
            if "labels" in inputs:
                labels = inputs["labels"]
                non_ignored = (labels != -100).sum().item()
                logger.debug(
                    f"Labels shape: {labels.shape}, non-100 count: {non_ignored}"
                )

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        if DEBUG:
            if loss is None:
                logger.error("LOSS IS NONE!")
            else:
                logger.debug(
                    f"Raw loss: {loss.item():.4f}, requires_grad: {loss.requires_grad}"
                )

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = False  # Set True for verbose debugging with 10 samples

# Model configuration - IT5 specific
MODEL_NAME = "gsarti/it5-large"  # Italian T5, 220M parameters
MODEL_TYPE = "it5"

# Training toggles
USE_LORA = True
USE_NEUTRALITY_GUIDANCE = False  # Set to True for classifier-guided loss
USE_GRADIENT_CHECKPOINTING = True  # Prevents OOM

# Paths
BASE_DIR = Path("/italian-hyperpartisan-neutralization")
TRAIN_FILE = BASE_DIR / "data/sft_train_original.jsonl"
DEV_FILE = BASE_DIR / "data/sft_dev_original.jsonl"
NEUTRALITY_CLASSIFIER_PATH = BASE_DIR / "outputs/models/bert_corrected/final_model"

# Sequence lengths
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

# IT5 NOTE: No task prefix needed - IT5 is not instruction-tuned like FLAN-T5
# The input will be: "{instruction}\n\n{input}" format
TASK_PREFIX = ""  # Empty for IT5

# Neutrality guidance parameters
GUIDANCE_WEIGHT = 2.0
GUIDANCE_FREQUENCY = 50
TARGET_NEUTRALITY_RATE = 0.85

# Training hyperparameters
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 2  # Fits on L4 GPU with LoRA
EVAL_BATCH_SIZE = 2
NUM_EPOCHS = 10 if not DEBUG else 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# Generation settings
GENERATION_NUM_BEAMS = 2  # Reduced for memory efficiency
GENERATION_MAX_LENGTH = MAX_TARGET_LENGTH

# Output directory
config_suffix = "CLASS" if USE_NEUTRALITY_GUIDANCE else "NOCLASS"
OUTPUT_DIR = BASE_DIR / f"outputs/models/sft/it5-base_lora_{config_suffix}"

# Logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_jsonl(file_path: Path) -> pd.DataFrame:
    """Load JSONL file into DataFrame."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return pd.DataFrame(data)


# =============================================================================
# TOKENIZER SETUP - IT5 SPECIFIC
# =============================================================================


def setup_tokenizer(model_name: str):
    """
    Load tokenizer for IT5.

    IT5 uses T5TokenizerFast which works with AutoTokenizer.
    Unlike FLAN-T5, we don't need T5Tokenizer with legacy=False.
    """
    # IT5 works with AutoTokenizer (returns T5TokenizerFast)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Loaded {type(tokenizer).__name__} for {model_name}")

    # Verify pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Set pad_token to eos_token")

    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    return tokenizer


# =============================================================================
# PREPROCESSING - IT5 SPECIFIC
# =============================================================================


def create_preprocess_function(
    tokenizer,
    max_input_length: int,
    max_target_length: int,
    task_prefix: str = "",
):
    """
    Create preprocessing function for IT5.

    IT5 differences from FLAN-T5:
    - No task prefix needed (IT5 is not instruction-tuned)
    - Use text_target parameter (works with T5TokenizerFast)
    - Still need to replace pad_token_id with -100 in labels
    """

    def preprocess_function(examples):
        # Combine instruction + input
        inputs = []
        for instr, inp in zip(examples["instruction"], examples["input"]):
            if task_prefix:
                text = f"{task_prefix}{inp}"
            else:
                # Standard seq2seq format for IT5
                text = f"{instr}\n\n{inp}"
            inputs.append(text)

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize targets using text_target (works with T5TokenizerFast)
        labels = tokenizer(
            text_target=examples["output"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

        # CRITICAL: Replace pad_token_id with -100 in labels
        # This ensures cross-entropy loss ignores padding positions
        label_ids = labels["input_ids"]
        label_ids_fixed = []
        for label_seq in label_ids:
            fixed_seq = [
                (token_id if token_id != tokenizer.pad_token_id else -100)
                for token_id in label_seq
            ]
            label_ids_fixed.append(fixed_seq)

        model_inputs["labels"] = label_ids_fixed

        return model_inputs

    return preprocess_function


def debug_tokenization(tokenizer, dataset, num_samples: int = 3):
    """Print sample tokenized examples for debugging."""
    logger.info("=" * 60)
    logger.info("DEBUG: Sample tokenized examples")
    logger.info("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Decode input
        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)

        # Decode labels (filter out -100)
        label_ids_valid = [l for l in labels if l != -100]
        label_text = tokenizer.decode(label_ids_valid, skip_special_tokens=False)

        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Input IDs shape: {len(input_ids)}")
        logger.info(f"Input (decoded): {input_text[:200]}...")
        logger.info(f"Labels shape: {len(labels)}")
        logger.info(f"Labels (non -100 count): {len(label_ids_valid)}")
        logger.info(f"Labels (decoded): {label_text[:200]}...")
        logger.info(f"First 10 label IDs: {labels[:10]}")

    logger.info("=" * 60)


# =============================================================================
# MODEL SETUP
# =============================================================================


def setup_model(model_name: str, use_gradient_checkpointing: bool = True):
    """
    Load IT5 model.

    IT5 uses the same architecture as T5, so we can use T5ForConditionalGeneration
    or AutoModelForSeq2SeqLM.
    """
    logger.info(f"Loading model: {model_name}")

    # Load model - IT5 is T5-based
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Set decoder_start_token_id if not set
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id or 0
        logger.info(
            f"Set decoder_start_token_id to {model.config.decoder_start_token_id}"
        )

    return model


def apply_lora(
    model, target_modules: List[str] = None, use_gradient_checkpointing: bool = True
):
    """
    Apply LoRA adapters to IT5 model.

    IT5 uses same architecture as T5, so same target modules work.
    """
    if target_modules is None:
        # T5/IT5 attention and feedforward layers
        target_modules = ["q", "k", "v", "o", "wi", "wo"]

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing AFTER LoRA
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("Enabled gradient checkpointing with use_reentrant=False")

    return model


# =============================================================================
# NEUTRALITY CLASSIFIER
# =============================================================================


class NeutralityClassifier:
    """Classifier to score generated text for neutrality."""

    def __init__(self, model_path: str, max_length: int = 256, batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size

        label2id = getattr(self.model.config, "label2id", {})
        self.neutral_id = label2id.get("neutral", 0)
        logger.info(f"NeutralityClassifier loaded, neutral_id={self.neutral_id}")

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {"neutrality_rate": 1.0}

        all_preds = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())

        neutrality_rate = sum(1 for p in all_preds if p == self.neutral_id) / len(
            all_preds
        )
        return {"neutrality_rate": neutrality_rate}


# =============================================================================
# METRICS
# =============================================================================

_bleu_metric = None
_rouge_metric = None


def _get_metrics():
    """Lazy load metrics."""
    global _bleu_metric, _rouge_metric
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("sacrebleu")
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _bleu_metric, _rouge_metric


def create_compute_metrics(tokenizer, vocab_size: int = None):
    """Create compute_metrics function with vocab overflow protection."""

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        bleu_metric, rouge_metric = _get_metrics()

        # Clip predictions to valid vocab range
        if vocab_size is not None:
            preds = np.clip(preds, 0, vocab_size - 1)

        # Replace NaN/Inf with pad token
        preds = np.nan_to_num(preds, nan=tokenizer.pad_token_id or 0)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels (handle -100)
        labels_cleaned = []
        for label in labels:
            label_clean = [l for l in label if l != -100]
            labels_cleaned.append(label_clean)
        decoded_labels = tokenizer.batch_decode(
            labels_cleaned, skip_special_tokens=True
        )

        if DEBUG and len(decoded_preds) > 0:
            logger.debug(f"Sample pred: {decoded_preds[0][:200]}")
            logger.debug(f"Sample label: {decoded_labels[0][:200]}")

        # Compute metrics
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds, references=[[ref] for ref in decoded_labels]
        )
        rouge_result = rouge_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        return {
            "bleu": bleu_result["score"],
            "rouge_l": rouge_result["rougeL"],
            "combined_score": bleu_result["score"] + rouge_result["rougeL"] * 100,
        }

    return compute_metrics


# =============================================================================
# CUSTOM TRAINER WITH NEUTRALITY GUIDANCE
# =============================================================================


class NeutralityGuidedTrainer(Seq2SeqTrainer):
    """Trainer with optional neutrality classifier guidance."""

    def __init__(
        self,
        *args,
        neutrality_classifier=None,
        guidance_weight=1.0,
        guidance_every_n_steps=5,
        guidance_target_rate=0.7,
        guidance_max_length=128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.neutrality_classifier = neutrality_classifier
        self.guidance_weight = guidance_weight
        self.guidance_every_n_steps = guidance_every_n_steps
        self.guidance_target_rate = guidance_target_rate
        self.guidance_max_length = guidance_max_length

        if self.neutrality_classifier is not None:
            self.neutrality_classifier.model.eval()
            for param in self.neutrality_classifier.model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with optional neutrality guidance."""
        outputs = model(**inputs)
        seq2seq_loss = outputs.loss

        should_guide = (
            self.neutrality_classifier is not None
            and model.training
            and self.state.global_step % self.guidance_every_n_steps == 0
            and self.guidance_weight > 0
        )

        if should_guide:
            try:
                neutrality_loss = self._compute_neutrality_loss(model, inputs)
                total_loss = seq2seq_loss + self.guidance_weight * neutrality_loss

                if DEBUG and self.state.global_step % 10 == 0:
                    logger.debug(
                        f"Step {self.state.global_step}: "
                        f"seq2seq_loss={seq2seq_loss.item():.4f}, "
                        f"neutrality_loss={neutrality_loss.item():.4f}"
                    )
            except Exception as e:
                logger.warning(f"Neutrality guidance failed: {e}")
                total_loss = seq2seq_loss
        else:
            total_loss = seq2seq_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_neutrality_loss(self, model, inputs):
        """Generate and score text for neutrality guidance."""
        gen_kwargs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "max_length": self.guidance_max_length,
            "num_beams": 1,  # Greedy for memory efficiency
            "do_sample": False,
        }

        with torch.no_grad():
            generated_ids = model.generate(**gen_kwargs)

        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        scores = self.neutrality_classifier(generated_texts)
        neutrality_rate = scores.get("neutrality_rate", 0.0)

        shortfall = max(0.0, self.guidance_target_rate - neutrality_rate)
        return torch.tensor(shortfall, device=inputs["input_ids"].device)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def main():
    """Main training entry point."""
    set_seed(42)

    logger.info("=" * 60)
    logger.info("IT5 Fine-tuning for Italian Text Neutralization")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"LoRA: {USE_LORA}")
    logger.info(f"Neutrality Guidance: {USE_NEUTRALITY_GUIDANCE}")
    logger.info(f"Gradient Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    logger.info(f"Debug Mode: {DEBUG}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = load_jsonl(TRAIN_FILE)
    dev_df = load_jsonl(DEV_FILE)

    # Debug mode
    if DEBUG:
        train_df = train_df.head(10)
        dev_df = dev_df.head(10)
        logger.info(f"DEBUG: Using {len(train_df)} train, {len(dev_df)} dev samples")

    # Setup tokenizer (IT5 uses AutoTokenizer)
    tokenizer = setup_tokenizer(MODEL_NAME)

    # Create preprocessing function
    preprocess_fn = create_preprocess_function(
        tokenizer=tokenizer,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        task_prefix=TASK_PREFIX,
    )

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df).map(
        preprocess_fn,
        batched=True,
        remove_columns=train_df.columns.tolist(),
        desc="Tokenizing train",
    )
    dev_dataset = Dataset.from_pandas(dev_df).map(
        preprocess_fn,
        batched=True,
        remove_columns=dev_df.columns.tolist(),
        desc="Tokenizing dev",
    )

    if DEBUG:
        debug_tokenization(tokenizer, train_dataset)

    # Setup model
    model = setup_model(
        MODEL_NAME, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )

    # Apply LoRA
    if USE_LORA:
        model = apply_lora(model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
    elif USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("Enabled gradient checkpointing on base model")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # Neutrality classifier
    neutrality_classifier = None
    if USE_NEUTRALITY_GUIDANCE and NEUTRALITY_CLASSIFIER_PATH.exists():
        neutrality_classifier = NeutralityClassifier(
            model_path=str(NEUTRALITY_CLASSIFIER_PATH),
            max_length=256,
            batch_size=8,
        )

    # Metrics
    compute_metrics = create_compute_metrics(tokenizer, vocab_size=tokenizer.vocab_size)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1 if DEBUG else 8,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,
        fp16=torch.cuda.is_available(),  # Use fp16 on GPU
        logging_steps=1 if DEBUG else 50,
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    # Create trainer
    callbacks_list = [EarlyStoppingCallback(early_stopping_patience=5)]
    if DEBUG:
        callbacks_list.append(DebugLossCallback())

    if USE_NEUTRALITY_GUIDANCE and neutrality_classifier is not None:
        trainer = NeutralityGuidedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks_list,
            neutrality_classifier=neutrality_classifier,
            guidance_weight=GUIDANCE_WEIGHT,
            guidance_every_n_steps=GUIDANCE_FREQUENCY,
            guidance_target_rate=TARGET_NEUTRALITY_RATE,
            guidance_max_length=min(128, MAX_TARGET_LENGTH),
        )
        logger.info("Using NeutralityGuidedTrainer with classifier guidance")
    else:
        trainer = FixedSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks_list,
        )
        logger.info("Using FixedSeq2SeqTrainer (no classifier)")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    trainer.save_model()
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Log metrics
    train_metrics = train_result.metrics
    if "train_loss" in train_metrics:
        train_metrics["train_perplexity"] = math.exp(
            min(train_metrics["train_loss"], 50)
        )
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    # Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Generate predictions
    logger.info("Generating predictions on dev set...")
    predictions = trainer.predict(
        dev_dataset,
        num_beams=GENERATION_NUM_BEAMS,
        max_length=GENERATION_MAX_LENGTH,
    )

    preds = predictions.predictions
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Save predictions
    output_df = dev_df.copy()
    output_df["predicted_output"] = decoded_preds
    output_file = OUTPUT_DIR / f"eval_generations_{config_suffix}.jsonl"
    output_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    logger.info(f"Predictions saved to {output_file}")

    # Print samples
    logger.info("\n" + "=" * 60)
    logger.info("SAMPLE PREDICTIONS")
    logger.info("=" * 60)
    for i in range(min(3, len(decoded_preds))):
        logger.info(f"\n--- Sample {i} ---")
        logger.info(f"Input: {dev_df.iloc[i]['input'][:200]}...")
        logger.info(f"Target: {dev_df.iloc[i]['output'][:200]}...")
        logger.info(f"Predicted: {decoded_preds[i][:200]}...")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

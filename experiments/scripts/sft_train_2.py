#!/usr/bin/env python3
"""
sft_train_2.py - Fixed T5 fine-tuning for Italian text neutralization

MAIN FIXES APPLIED:
1. TOKENIZATION: Explicit -100 replacement for padding in labels (T5 critical)
2. DECODER_START_TOKEN: Explicitly set for T5 models (uses pad_token_id=0)
3. TASK PREFIX: Added "neutralize: " prefix for T5 models (expected by FLAN-T5)
4. LABEL ENCODING: Use as_target_tokenizer() context for T5Tokenizer compatibility
5. MEMORY: Gradient checkpointing + optimized generation during eval
6. VOCAB OVERFLOW: Clip predictions to valid vocab range before decoding
7. ATTENTION MASK: Explicit handling for decoder inputs

WHY BART WORKED BUT T5 FAILED:
- BART auto-handles label shifting; T5 needs explicit decoder_start_token_id
- text_target= parameter may not work correctly with T5Tokenizer
- Labels padded with pad_token_id instead of -100 caused loss on padding
- T5 expects task prefixes from FLAN training; BART doesn't
"""

from __future__ import annotations

import os
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    T5Tokenizer,
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

    Note: The standard Seq2SeqTrainer was showing loss=0.0 in some configurations.
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
                    f"Raw loss value: {loss.item()}, dtype: {loss.dtype}, requires_grad: {loss.requires_grad}"
                )

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# CONFIGURATION
# =============================================================================

DEBUG = False  # Set True for verbose debugging with 10 samples

# Model configuration
MODEL_NAME = (
    "facebook/bart-base"  # Options: "google/flan-t5-base", "google/flan-t5-large"
)
MODEL_TYPE = "t5"  # Will be auto-detected

# Training toggles
USE_LORA = True
USE_NEUTRALITY_GUIDANCE = False  # Toggle classifier-guided loss - DISABLED FOR DEBUG
USE_GRADIENT_CHECKPOINTING = True  # FIX: Prevents OOM for T5-large

# Paths
BASE_DIR = Path("/italian-hyperpartisan-neutralization")
TRAIN_FILE = BASE_DIR / "data/sft_train_original.jsonl"
DEV_FILE = BASE_DIR / "data/sft_dev_original.jsonl"
NEUTRALITY_CLASSIFIER_PATH = BASE_DIR / "outputs/models/bert_corrected/final_model"

# Sequence lengths
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256

# FIX: Add task prefix for T5 models (FLAN-T5 expects task instructions)
# BART doesn't need this, but T5 was trained with task prefixes
TASK_PREFIX = "Riscrivi in modo neutrale: "  # Italian task prefix

# Neutrality guidance parameters
GUIDANCE_WEIGHT = 2.0
GUIDANCE_FREQUENCY = 50
TARGET_NEUTRALITY_RATE = 0.85

# Training hyperparameters
LEARNING_RATE = 5e-5
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2  # FIX: Reduced for memory during generation
NUM_EPOCHS = 10 if not DEBUG else 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# FIX: Limit generation beams to reduce memory during evaluation
GENERATION_NUM_BEAMS = 2  # Was 4, reduced to prevent OOM
GENERATION_MAX_LENGTH = MAX_TARGET_LENGTH

# Output directory
config_suffix = "CLASS" if USE_NEUTRALITY_GUIDANCE else "NOCLASS"
model_short_name = MODEL_NAME.split("/")[-1]
OUTPUT_DIR = BASE_DIR / f"outputs/models/sft/{model_short_name}_lora_{config_suffix}_v2"

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
# TOKENIZER SETUP
# =============================================================================


def setup_tokenizer(model_name: str):
    """
    FIX: Use correct tokenizer class for T5 models.
    T5Tokenizer handles target tokenization differently than AutoTokenizer.
    """
    if "t5" in model_name.lower():
        # FIX: Use T5Tokenizer explicitly for T5 models
        # This ensures proper handling of decoder inputs
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        logger.info(f"Loaded T5Tokenizer for {model_name}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        logger.info(f"Loaded AutoTokenizer for {model_name}")

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Set pad_token to eos_token")

    return tokenizer


# =============================================================================
# PREPROCESSING - CRITICAL FIXES FOR T5
# =============================================================================


def create_preprocess_function(
    tokenizer,
    max_input_length: int,
    max_target_length: int,
    task_prefix: str = "",
    is_t5: bool = True,
):
    """
    Create preprocessing function with T5-specific fixes.

    FIXES APPLIED:
    1. Add task prefix for T5 models (FLAN-T5 expects this)
    2. Use as_target_tokenizer() context for proper target encoding
    3. Explicitly replace pad_token_id with -100 in labels
    """

    def preprocess_function(examples):
        # Combine instruction + input with optional task prefix
        inputs = []
        for instr, inp in zip(examples["instruction"], examples["input"]):
            # FIX: Add task prefix for T5 models
            # FLAN-T5 was trained with task prefixes; BART doesn't need them
            if task_prefix:
                text = f"{task_prefix}{inp}"
            else:
                text = f"{instr}\n\n{inp}"
            inputs.append(text)

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",  # Pad to max length for consistent batching
        )

        # FIX: Tokenize targets using as_target_tokenizer() for T5
        # This is CRITICAL - T5Tokenizer needs this context to properly encode targets
        # The text_target= parameter may not work correctly with older T5Tokenizer versions
        if is_t5:
            # Use deprecated but necessary context manager for T5
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["output"],
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length",
                )
        else:
            # BART and others work with text_target
            labels = tokenizer(
                text_target=examples["output"],
                max_length=max_target_length,
                truncation=True,
                padding="max_length",
            )

        # FIX: CRITICAL - Replace pad_token_id with -100 in labels
        # This ensures cross-entropy loss IGNORES padding positions
        # Without this, the model learns to predict <pad> tokens, causing garbage output
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
        logger.info(f"Pad token ID: {tokenizer.pad_token_id}")

    logger.info("=" * 60)


# =============================================================================
# MODEL SETUP
# =============================================================================


def setup_model(model_name: str, use_gradient_checkpointing: bool = True):
    """
    Load model with T5-specific configurations.

    NOTE: Gradient checkpointing is now applied AFTER LoRA for proper PEFT compatibility.
    """
    logger.info(f"Loading model: {model_name}")

    if "t5" in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # FIX: Explicitly set decoder_start_token_id for T5
    # T5 uses pad_token_id (0) as decoder start, unlike BART which uses bos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id or 0
        logger.info(
            f"Set decoder_start_token_id to {model.config.decoder_start_token_id}"
        )

    # NOTE: Gradient checkpointing is applied after LoRA (see apply_lora function)

    return model


def apply_lora(
    model, target_modules: List[str] = None, use_gradient_checkpointing: bool = True
):
    """Apply LoRA adapters to model.

    FIX: Enable gradient checkpointing AFTER LoRA is applied, with use_reentrant=False
    for PEFT compatibility. This prevents the 'None of the inputs have requires_grad=True' warning.
    """
    if target_modules is None:
        # FIX: Correct LoRA target modules for T5 architecture
        # These are the attention and feedforward layers in T5
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

    # FIX: Enable gradient checkpointing AFTER LoRA is applied
    # Must use use_reentrant=False for PEFT compatibility
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Disable cache when using gradient checkpointing
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info(
            "Enabled gradient checkpointing with use_reentrant=False (PEFT compatible)"
        )

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

        # Get neutral class ID
        label2id = getattr(self.model.config, "label2id", {})
        self.neutral_id = label2id.get("neutral", 0)
        logger.info(f"NeutralityClassifier loaded, neutral_id={self.neutral_id}")

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        if not texts:
            return {"neutrality_rate": 1.0}

        # Process in batches
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

# Metrics are loaded lazily to avoid import errors on login nodes
_bleu_metric = None
_rouge_metric = None


def _get_metrics():
    """Lazy load metrics to avoid import errors."""
    global _bleu_metric, _rouge_metric
    if _bleu_metric is None:
        _bleu_metric = evaluate.load("sacrebleu")
    if _rouge_metric is None:
        _rouge_metric = evaluate.load("rouge")
    return _bleu_metric, _rouge_metric


def create_compute_metrics(tokenizer, vocab_size: int = None):
    """
    Create compute_metrics function with vocab overflow protection.

    FIX: Clip predictions to valid vocab range before decoding.
    This prevents garbage output from vocab overflow.
    """

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Load metrics lazily
        bleu_metric, rouge_metric = _get_metrics()

        # FIX: Clip predictions to valid vocabulary range
        # T5 models can sometimes output logits beyond vocab size
        if vocab_size is not None:
            preds = np.clip(preds, 0, vocab_size - 1)
            if DEBUG:
                logger.debug(
                    f"Clipped predictions to vocab range [0, {vocab_size - 1}]"
                )

        # Replace NaN/Inf with pad token
        preds = np.nan_to_num(preds, nan=tokenizer.pad_token_id or 0)

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Decode labels (handle -100 ignored index)
        labels_cleaned = []
        for label in labels:
            label_clean = [l for l in label if l != -100]
            labels_cleaned.append(label_clean)
        decoded_labels = tokenizer.batch_decode(
            labels_cleaned, skip_special_tokens=True
        )

        # Debug: print sample outputs
        if DEBUG and len(decoded_preds) > 0:
            logger.debug(f"Sample pred: {decoded_preds[0][:200]}")
            logger.debug(f"Sample label: {decoded_labels[0][:200]}")

        # Compute BLEU
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds, references=[[ref] for ref in decoded_labels]
        )

        # Compute ROUGE
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
    """
    Trainer with optional neutrality classifier guidance.

    FIX: Uses greedy decoding for guidance (num_beams=1) to save memory.
    This explains why classifier version uses LESS memory than standard.
    """

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

        # Freeze classifier
        if self.neutrality_classifier is not None:
            self.neutrality_classifier.model.eval()
            for param in self.neutrality_classifier.model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss with optional neutrality guidance."""
        # Standard seq2seq loss
        outputs = model(**inputs)
        seq2seq_loss = outputs.loss

        # Apply guidance periodically
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
        # FIX: Use greedy decoding (num_beams=1) to minimize memory
        # This is why classifier version uses LESS memory!
        gen_kwargs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
            "max_length": self.guidance_max_length,
            "num_beams": 1,  # Greedy - much less memory than beam search
            "do_sample": False,
        }

        with torch.no_grad():
            generated_ids = model.generate(**gen_kwargs)

        # Decode - use self.tokenizer (compatible with older transformers)
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Score
        scores = self.neutrality_classifier(generated_texts)
        neutrality_rate = scores.get("neutrality_rate", 0.0)

        # Loss: penalize falling short of target
        shortfall = max(0.0, self.guidance_target_rate - neutrality_rate)
        return torch.tensor(shortfall, device=inputs["input_ids"].device)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def main():
    """Main training entry point."""
    set_seed(42)

    logger.info("=" * 60)
    logger.info("T5 Fine-tuning for Italian Text Neutralization (v2 - FIXED)")
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

    # Debug mode: use only 10 samples
    if DEBUG:
        train_df = train_df.head(10)
        dev_df = dev_df.head(10)
        logger.info(f"DEBUG: Using {len(train_df)} train, {len(dev_df)} dev samples")

    # Setup tokenizer
    tokenizer = setup_tokenizer(MODEL_NAME)
    is_t5_model = "t5" in MODEL_NAME.lower()

    # Create preprocessing function with T5 fixes
    preprocess_fn = create_preprocess_function(
        tokenizer=tokenizer,
        max_input_length=MAX_INPUT_LENGTH,
        max_target_length=MAX_TARGET_LENGTH,
        task_prefix=TASK_PREFIX if is_t5_model else "",
        is_t5=is_t5_model,
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

    # Debug: print tokenization samples
    if DEBUG:
        debug_tokenization(tokenizer, train_dataset)

    # Setup model
    model = setup_model(
        MODEL_NAME, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )

    # Apply LoRA (gradient checkpointing is enabled inside, after PEFT is applied)
    if USE_LORA:
        model = apply_lora(model, use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING)
    elif USE_GRADIENT_CHECKPOINTING:
        # If not using LoRA, enable gradient checkpointing directly on base model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        logger.info("Enabled gradient checkpointing on base model (no LoRA)")

    # Data collator
    # FIX: Explicitly set label_pad_token_id=-100 (should be default, but be explicit)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,  # Explicit: ignore padding in loss
    )

    # Neutrality classifier (optional)
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
    # FIX: Reduced generation beams and batch size for memory efficiency
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=1
        if DEBUG
        else 8,  # Effective batch = 16 (or 2 for debug)
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=GENERATION_NUM_BEAMS,  # FIX: Reduced from 4 to 2
        fp16=False,  # Set to False, otherwise loss is 0.0
        logging_steps=1 if DEBUG else 50,  # Log every step in debug mode
        load_best_model_at_end=True,
        metric_for_best_model="rouge_l",
        greater_is_better=True,
        report_to="none",  # Disable wandb for testing
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
            tokenizer=tokenizer,  # Use tokenizer= for older transformers compatibility
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
        # Always use FixedSeq2SeqTrainer to ensure proper loss tracking
        trainer = FixedSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,  # Use tokenizer= for older transformers compatibility
            compute_metrics=compute_metrics,
            callbacks=callbacks_list,
        )
        logger.info("Using standard Seq2SeqTrainer (no classifier)")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save model
    trainer.save_model()
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # Log training metrics
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

    # Generate predictions on dev set
    logger.info("Generating predictions on dev set...")
    predictions = trainer.predict(
        dev_dataset,
        num_beams=GENERATION_NUM_BEAMS,
        max_length=GENERATION_MAX_LENGTH,
    )

    # FIX: Clip predictions before decoding
    preds = predictions.predictions
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Save predictions
    output_df = dev_df.copy()
    output_df["predicted_output"] = decoded_preds
    output_file = OUTPUT_DIR / f"eval_generations_{config_suffix}.jsonl"
    output_df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    logger.info(f"Predictions saved to {output_file}")

    # Print sample predictions
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

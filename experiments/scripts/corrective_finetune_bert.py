#!/usr/bin/env python
"""Continue fine-tuning the Italian BERT classifier on corrective data (manual training loop)."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (  # type: ignore
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

DEFAULT_CLASS_WEIGHTS = (1.0, 2.0)  # (hyperpartisan, neutral)
DEFAULT_LABEL_MAPPING = {"neutral": 0, "hyperpartisan": 1}


class CorrectiveDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrective fine-tuning for Italian BERT classifier"
    )
    parser.add_argument(
        "--train_file", type=Path, default=Path("data/corrective_train.csv")
    )
    parser.add_argument(
        "--val_file", type=Path, default=Path("data/corrective_val.csv")
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/italian-hyperpartisan-neutralization/XAI_HIPP/models/FT/sent-ita-xxl",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("outputs/models/bert_corrected")
    )
    parser.add_argument(
        "--best_model_dir",
        type=Path,
        default=Path("outputs/models/bert_corrected/best"),
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--class_weight_hyper", type=float, default=DEFAULT_CLASS_WEIGHTS[0]
    )
    parser.add_argument(
        "--class_weight_neutral", type=float, default=DEFAULT_CLASS_WEIGHTS[1]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--log_file",
        type=Path,
        default=Path("outputs/models/bert_corrected/training_logs.json"),
    )
    parser.add_argument(
        "--report_file",
        type=Path,
        default=Path("outputs/models/bert_corrected/before_after_comparison.txt"),
    )
    parser.add_argument(
        "--metrics_file",
        type=Path,
        default=Path("outputs/models/bert_corrected/final_metrics.json"),
    )
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=Path("outputs/models/bert_corrected/val_predictions.csv"),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional limit on training samples for debugging",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional limit on eval samples for debugging",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if {"text", "label"} - set(df.columns):
        raise ValueError(f"Dataset {path} must contain 'text' and 'label' columns")
    return df.dropna(subset=["text", "label"]).copy()


def map_labels(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    label_map = {
        "neutral": DEFAULT_LABEL_MAPPING["neutral"],
        "hyperpartisan": DEFAULT_LABEL_MAPPING["hyperpartisan"],
        "0": DEFAULT_LABEL_MAPPING["neutral"],
        "1": DEFAULT_LABEL_MAPPING["hyperpartisan"],
    }
    texts: List[str] = []
    labels: List[int] = []
    for _, row in df.iterrows():
        raw_label = row["label"]
        if isinstance(raw_label, str):
            key = raw_label.strip().lower()
        else:
            key = str(int(raw_label))
        if key not in label_map:
            raise ValueError(
                f"Unknown label '{raw_label}'. Expected neutral/hyperpartisan or 0/1."
            )
        text = str(row["text"]).strip()
        if not text:
            continue
        texts.append(text)
        labels.append(label_map[key])
    return texts, labels


def build_dataloader(
    texts: List[str],
    labels: List[int],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CorrectiveDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    label_names: List[str],
) -> Tuple[Dict[str, Any], List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    refs: List[int] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())
            refs.extend(labels.cpu().tolist())
    bal_acc = balanced_accuracy_score(refs, preds)
    acc = accuracy_score(refs, preds)
    report = classification_report(
        refs,
        preds,
        target_names=label_names,
        labels=list(range(len(label_names))),
        output_dict=True,
        zero_division=0,
    )
    report_dict = cast(Dict[str, Dict[str, float]], report)
    metrics = {
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "macro_f1": report_dict["macro avg"]["f1-score"],
    }
    for name in label_names:
        metrics[f"precision_{name}"] = report_dict[name]["precision"]
        metrics[f"recall_{name}"] = report_dict[name]["recall"]
        metrics[f"f1_{name}"] = report_dict[name]["f1-score"]
    return metrics, preds, refs


def format_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


def write_comparison(
    pre_metrics: Dict[str, float], post_metrics: Dict[str, float], path: Path
) -> None:
    keys = [
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "recall_hyperpartisan",
        "recall_neutral",
        "precision_hyperpartisan",
        "precision_neutral",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("Before vs After Evaluation\n")
        fp.write("Metric\tBefore\tAfter\tDelta\n")
        for key in keys:
            before = pre_metrics.get(f"pre_{key}", float("nan"))
            after = post_metrics.get(f"post_{key}", float("nan"))
            if math.isnan(before) or math.isnan(after):
                fp.write(f"{key}\t{before}\t{after}\tNaN\n")
                continue
            delta = after - before
            fp.write(f"{key}\t{before:.4f}\t{after:.4f}\t{delta:+.4f}\n")


def maybe_limit(
    texts: List[str], labels: List[int], limit: int | None, seed: int
) -> Tuple[List[str], List[int]]:
    if not limit or limit >= len(texts):
        return texts, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(texts), size=limit, replace=False)
    indices = sorted(indices)
    new_texts = [texts[i] for i in indices]
    new_labels = [labels[i] for i in indices]
    return new_texts, new_labels


def save_predictions(
    path: Path,
    texts: List[str],
    refs: List[int],
    preds: List[int],
    label_names: List[str],
) -> None:
    if not (len(texts) == len(refs) == len(preds)):
        raise ValueError("Prediction export mismatch between texts and labels")
    records = []
    for text, ref, pred in zip(texts, refs, preds):
        records.append(
            {
                "text": text,
                "label_id": ref,
                "label_name": label_names[ref],
                "prediction_id": pred,
                "prediction_name": label_names[pred],
                "is_correct": bool(ref == pred),
            }
        )
    df = pd.DataFrame.from_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def training_loop(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = read_dataset(args.train_file)
    val_df = read_dataset(args.val_file)
    train_texts, train_labels = map_labels(train_df)
    val_texts, val_labels = map_labels(val_df)

    train_texts, train_labels = maybe_limit(
        train_texts, train_labels, args.max_train_samples, args.seed
    )
    val_texts, val_labels = maybe_limit(
        val_texts, val_labels, args.max_eval_samples, args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "right"

    train_loader = build_dataloader(
        train_texts,
        train_labels,
        tokenizer,
        args.max_length,
        args.per_device_train_batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_texts,
        val_labels,
        tokenizer,
        args.max_length,
        args.per_device_eval_batch_size,
        shuffle=False,
    )

    label_names = ["neutral", "hyperpartisan"]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "neutral", 1: "hyperpartisan"},
        label2id={"neutral": 0, "hyperpartisan": 1},
    )
    model.to(device)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = (
        math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        * args.num_train_epochs
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    weights = torch.tensor(
        [args.class_weight_neutral, args.class_weight_hyper],
        dtype=torch.float,
        device=device,
    )
    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    os.makedirs(args.output_dir, exist_ok=True)
    training_history: List[Dict[str, Any]] = []

    pre_metrics_raw, _, _ = evaluate_model(model, val_loader, device, label_names)
    pre_metrics = format_metrics(pre_metrics_raw, "pre")
    best_metric = pre_metrics_raw.get("balanced_accuracy", 0.0)
    checkpoint_root = args.output_dir / "checkpoints"
    saved_checkpoints: List[Path] = []
    best_checkpoint_path: Path | None = None

    global_step = 0
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = loss_fct(outputs.logits, labels)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                training_history.append(
                    {
                        "event": "train_step",
                        "epoch": epoch,
                        "step": global_step,
                        "loss": running_loss,
                    }
                )
                running_loss = 0.0
        post_epoch_metrics_raw, _, _ = evaluate_model(
            model, val_loader, device, label_names
        )
        best_check = float(post_epoch_metrics_raw["balanced_accuracy"])
        if best_check >= best_metric:
            best_metric = best_check
            save_dir = checkpoint_root / f"epoch{epoch:02d}"
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            saved_checkpoints.append(save_dir)
            best_checkpoint_path = save_dir
            while len(saved_checkpoints) > args.save_total_limit:
                old = saved_checkpoints.pop(0)
                shutil.rmtree(old, ignore_errors=True)
        training_history.append(
            {
                "event": "eval_epoch",
                "epoch": epoch,
                **{f"val_{k}": float(v) for k, v in post_epoch_metrics_raw.items()},
            }
        )

    final_metrics_raw, final_preds, final_refs = evaluate_model(
        model, val_loader, device, label_names
    )
    final_metrics = format_metrics(final_metrics_raw, "post")

    if best_checkpoint_path and best_checkpoint_path.exists():
        if args.best_model_dir.exists():
            shutil.rmtree(args.best_model_dir)
        shutil.copytree(best_checkpoint_path, args.best_model_dir)
    else:
        args.best_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.best_model_dir)
        tokenizer.save_pretrained(args.best_model_dir)

    final_model_dir = args.output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as fp:
        json.dump(training_history, fp, indent=2)

    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "pre": pre_metrics_raw,
        "post": final_metrics_raw,
        "best_balanced_accuracy": best_metric,
        "run_config": {
            "train_examples": len(train_texts),
            "val_examples": len(val_texts),
            "num_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "base_model": args.base_model,
        },
    }
    with args.metrics_file.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    save_predictions(
        args.predictions_file, val_texts, final_refs, final_preds, label_names
    )
    write_comparison(pre_metrics, final_metrics, args.report_file)


def main() -> None:
    args = parse_args()
    try:
        training_loop(args)
    except Exception as exc:  # pragma: no cover - CLI entry point
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

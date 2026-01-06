"""Custom Trainer callbacks used for SFT training."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from datasets import Dataset
from evaluate import load as load_metric
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover - fallback when wandb is unavailable
    wandb = None


def _log_to_wandb(payload: Mapping[str, Any]) -> None:
    """Safely log to Weights & Biases if it is enabled."""
    if wandb is not None and wandb.run is not None:  # pragma: no branch
        wandb.log(dict(payload))


def _last_logged_train_loss(state: TrainerState) -> Optional[float]:
    """Fetch the latest train loss from the Trainer state history."""
    for entry in reversed(state.log_history):
        if "loss" in entry:
            return entry.get("loss")
        if "train_loss" in entry:
            return entry.get("train_loss")
    return None


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    num_beams: int = 4,
) -> str:
    """Generate a single sequence from the current model."""
    device = model.device
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_length=max_length, num_beams=num_beams
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


class OverfittingDetectionCallback(TrainerCallback):
    """Detects and surfaces basic overfitting signals."""

    def __init__(self, warning_gap: float = 0.5) -> None:
        self.warning_gap = warning_gap

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        train_loss = _last_logged_train_loss(state)
        eval_loss = metrics.get("eval_loss")
        if train_loss is None or eval_loss is None:
            return
        gap = eval_loss - train_loss
        payload = {
            "overfitting/train_dev_gap": gap,
            "overfitting/train_loss": train_loss,
            "overfitting/eval_loss": eval_loss,
            "step": state.global_step,
        }
        _log_to_wandb(payload)
        if gap > self.warning_gap:
            print(f"⚠️  WARNING: Large train-dev gap detected: {gap:.3f}")


class GenerationSamplerCallback(TrainerCallback):
    """Log qualitative generations from train/dev splits during training."""

    def __init__(
        self,
        tokenizer,
        sample_datasets: Mapping[str, Dataset],
        num_samples: int = 5,
        sample_every: int = 1000,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        log_path: Optional[Path] = None,
        input_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sample_datasets = sample_datasets
        self.num_samples = num_samples
        self.sample_every = sample_every
        self.generation_kwargs = generation_kwargs or {
            "max_length": 512,
            "num_beams": 4,
        }
        self._last_step_logged = -1
        self.log_path = log_path
        self._fh = None
        self._input_builder = input_builder
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open("a", encoding="utf-8")

    def _prepare_samples(self, dataset: Dataset) -> List[Dict[str, str]]:
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        chosen = idxs[: self.num_samples]
        return [dataset[int(i)] for i in chosen]

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        if state.global_step - self._last_step_logged < self.sample_every:
            return
        self._last_step_logged = state.global_step
        for split_name, dataset in self.sample_datasets.items():
            samples = self._prepare_samples(dataset)
            table_rows: List[List[str]] = []
            for sample in samples:
                input_text = sample.get("input", "")
                target = sample.get("output", "")
                if self._input_builder is not None:
                    prompt = self._input_builder(sample)
                else:
                    prompt = sample.get("instruction", "").strip()
                    if prompt:
                        prompt = f"{prompt}\n\n{input_text}"
                    else:
                        prompt = input_text
                generated = generate_text(
                    model,
                    self.tokenizer,
                    prompt,
                    max_length=self.generation_kwargs.get("max_length", 512),
                    num_beams=self.generation_kwargs.get("num_beams", 4),
                )
                table_rows.append([input_text[:200], target[:200], generated[:200]])
                self._write_sample(
                    split_name,
                    state.global_step,
                    sample,
                    generated,
                )
            if table_rows:
                if wandb is not None and wandb.run is not None:  # type: ignore[attr-defined]
                    table = wandb.Table(  # type: ignore[attr-defined]
                        columns=["Input", "Target", "Generated"], data=table_rows
                    )
                    _log_to_wandb(
                        {f"samples/{split_name}": table, "step": state.global_step}
                    )
                else:
                    print(
                        f"Sample generations for {split_name} at step {state.global_step}"
                    )

    def _write_sample(
        self,
        split_name: str,
        step: int,
        sample: Mapping[str, Any],
        generated: str,
    ) -> None:
        if self._fh is None:
            return
        model_input = None
        if self._input_builder is not None:
            model_input = self._input_builder(sample)
        record = {
            "timestamp": time.time(),
            "step": step,
            "split": split_name,
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "target": sample.get("output", ""),
            "entry_id": sample.get("entry_id"),
            "generated": generated,
            "model_input": model_input,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            if self._fh is not None and not self._fh.closed:
                self._fh.close()
        except Exception:
            pass


class SampleBleuCallback(TrainerCallback):
    """Compute BLEU on small samples from train/dev to monitor overfitting."""

    def __init__(
        self,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        sample_size: int = 32,
        eval_interval: int = 5,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        input_builder: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.sample_size = sample_size
        self.eval_interval = eval_interval
        self._eval_calls = 0
        self._bleu = load_metric("sacrebleu")
        self.generation_kwargs = generation_kwargs or {
            "max_length": 512,
            "num_beams": 4,
        }
        self._input_builder = input_builder

    def _create_prompt(self, example: Mapping[str, str]) -> str:
        if self._input_builder is not None:
            return self._input_builder(example)
        prompt = example.get("instruction", "").strip()
        input_text = example.get("input", "")
        if prompt and input_text:
            return f"{prompt}\n\n{input_text}"
        return prompt or input_text

    def _compute_sample_bleu(self, dataset: Dataset, model: torch.nn.Module) -> float:
        if len(dataset) == 0:
            return float("nan")
        indices = random.sample(
            range(len(dataset)), k=min(self.sample_size, len(dataset))
        )
        predictions: List[str] = []
        references: List[List[str]] = []
        for idx in indices:
            example = dataset[int(idx)]
            prompt = self._create_prompt(example)
            generated = generate_text(
                model,
                self.tokenizer,
                prompt,
                max_length=self.generation_kwargs.get("max_length", 512),
                num_beams=self.generation_kwargs.get("num_beams", 4),
            )
            predictions.append(generated)
            references.append([example.get("output", "")])
        results = self._bleu.compute(predictions=predictions, references=references)
        return float(results.get("score", 0.0))

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        self._eval_calls += 1
        if self._eval_calls % self.eval_interval != 0:
            return
        train_bleu = self._compute_sample_bleu(self.train_dataset, model)
        dev_bleu = self._compute_sample_bleu(self.eval_dataset, model)
        payload = {
            "train_bleu_sample": train_bleu,
            "eval_bleu_sample": dev_bleu,
            "step": state.global_step,
        }
        _log_to_wandb(payload)


class TrainingStatsCallback(TrainerCallback):
    """Log gradient stats early and dump diagnostics when anomalies appear."""

    def __init__(
        self,
        log_interval: int = 50,
        log_first_n_steps: int = 0,
        log_path: Optional[Path] = None,
        batch_recorder: Optional[Any] = None,
    ) -> None:
        self.log_interval = max(1, int(log_interval))
        self.log_first_n_steps = max(0, int(log_first_n_steps))
        self._optimizer = None
        self.log_path = log_path
        self.batch_recorder = batch_recorder
        self._fh = None
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.log_path.open("a", encoding="utf-8")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]

    def on_step_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: torch.nn.Module,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step == 0:
            return
        should_log = False
        if self.log_first_n_steps and step <= self.log_first_n_steps:
            should_log = True
        elif self.log_interval <= 1 or step % self.log_interval == 0:
            should_log = True
        if not should_log:
            return
        grad_norms: List[float] = []
        grad_has_nan = False
        grad_has_inf = False
        anomalous_params: List[Dict[str, Any]] = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_tensor = param.grad.detach()
            grad_float = grad_tensor.float()
            grad_norm = grad_float.norm(2)
            grad_norms.append(float(grad_norm.cpu().item()))
            has_nan = torch.isnan(grad_tensor).any().item()
            has_inf = torch.isinf(grad_tensor).any().item()
            if has_nan:
                grad_has_nan = True
            if has_inf:
                grad_has_inf = True
            if has_nan or has_inf:
                finite_vals = grad_tensor[torch.isfinite(grad_tensor)]
                grad_min = (
                    float(finite_vals.min().item()) if finite_vals.numel() else math.nan
                )
                grad_max = (
                    float(finite_vals.max().item()) if finite_vals.numel() else math.nan
                )
                anomalous_params.append(
                    {
                        "name": name,
                        "shape": list(grad_tensor.shape),
                        "has_nan": bool(has_nan),
                        "has_inf": bool(has_inf),
                        "finite_min": grad_min,
                        "finite_max": grad_max,
                    }
                )
        if not grad_norms:
            return
        total_norm_tensor = torch.tensor(grad_norms, dtype=torch.float32)
        total_norm = float(torch.norm(total_norm_tensor, p=2).item())
        lr = math.nan
        if self._optimizer is not None and self._optimizer.param_groups:
            lr = self._optimizer.param_groups[0].get("lr", math.nan)
        payload = {
            "grad_norm": total_norm,
            "learning_rate": lr,
            "step": step,
        }
        if grad_has_nan:
            payload["grad_has_nan"] = True
        if grad_has_inf:
            payload["grad_has_inf"] = True
        anomaly_detected = (
            grad_has_nan
            or grad_has_inf
            or math.isnan(total_norm)
            or math.isinf(total_norm)
        )
        recent_batches: Optional[List[Dict[str, Any]]] = None
        if anomaly_detected and self.batch_recorder is not None:
            recent_batches = self.batch_recorder.latest(4)
        _log_to_wandb(payload)
        if self._fh is not None:
            record = {
                "timestamp": time.time(),
                **payload,
                "anomalous_parameters": anomalous_params,
            }
            if recent_batches:
                record["recent_batches"] = recent_batches
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


class FileLoggingCallback(TrainerCallback):
    """Persist trainer metrics to a JSONL file for offline analysis."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def _write(self, event: str, step: int, metrics: Mapping[str, Any]) -> None:
        record = {
            "event": event,
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def on_log(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        self._write("log", state.global_step, logs)

    def on_evaluate(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        self._write("evaluate", state.global_step, metrics)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._write("train_end", state.global_step, {"best_metric": state.best_metric})

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


class NaNSafeguardCallback(TrainerCallback):
    """Abort training as soon as NaN/Inf metrics are observed."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        batch_recorder: Optional[Any] = None,
        diag_log_path: Optional[Path] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.batch_recorder = batch_recorder
        self._diag_fh = None
        if diag_log_path is not None:
            diag_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._diag_fh = diag_log_path.open("a", encoding="utf-8")

    def _emit_diag(self, metric: str, value: float, step: int) -> None:
        if self._diag_fh is None or self.batch_recorder is None:
            return
        recent_batches = self.batch_recorder.latest(4)
        if not recent_batches:
            return
        record = {
            "timestamp": time.time(),
            "metric": metric,
            "value": value,
            "step": step,
            "recent_batches": recent_batches,
        }
        self._diag_fh.write(json.dumps(record) + "\n")
        self._diag_fh.flush()

    def on_log(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        for key, value in logs.items():
            if not isinstance(value, Number):
                continue
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                self.logger.error(
                    "Detected invalid metric %s=%s at step %s; stopping training",
                    key,
                    val,
                    state.global_step,
                )
                self._emit_diag(key, val, state.global_step)
                control.should_training_stop = True
                control.should_early_stop = True
                raise RuntimeError(f"NaN/Inf detected in metric '{key}'")

    def __del__(self) -> None:  # pragma: no cover - cleanup helper
        try:
            if self._diag_fh is not None and not self._diag_fh.closed:
                self._diag_fh.close()
        except Exception:
            pass


__all__ = [
    "OverfittingDetectionCallback",
    "GenerationSamplerCallback",
    "SampleBleuCallback",
    "TrainingStatsCallback",
    "FileLoggingCallback",
    "NaNSafeguardCallback",
    "generate_text",
]

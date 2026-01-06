"""
DPO (Direct Preference Optimization) Training Script.

Finetunes seq2seq models using TRL's DPOTrainer with preference pairs.
Data format: {"prompt": str, "chosen": str, "rejected": str}
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, DatasetDict
from evaluate import load as load_metric
from rouge_score import rouge_scorer
import sacrebleu
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
)
from bert_score import score as bertscore_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from comet import download_model as comet_download_model
    from comet import load_from_checkpoint as comet_load_from_checkpoint
except ImportError:  # pragma: no cover - optional dependency
    comet_download_model = None
    comet_load_from_checkpoint = None

_SENTENCE_TRANSFORMER = None
_SENTENCE_TRANSFORMER_NAME: Optional[str] = None
_COMET_MODEL = None
_COMET_MODEL_NAME: Optional[str] = None

# TRL for DPO
try:
    from trl import DPOConfig, DPOTrainer
except ImportError:
    print("TRL not installed. Install with: pip install trl")
    sys.exit(1)


def _ensure_tensor(value, device: torch.device, name: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    original_type = type(value)
    # unwrap torch.return_types.* objects (e.g., torch.max returns)
    if hasattr(value, "values") and isinstance(value.values, torch.Tensor):
        return value.values.to(device)
    if hasattr(value, "_fields"):
        first_field = getattr(value, value._fields[0], None)
        if isinstance(first_field, torch.Tensor):
            return first_field.to(device)
    if isinstance(value, (list, tuple)):
        if not value:
            tensor = torch.empty(0, dtype=torch.float32, device=device)
        elif isinstance(value[0], torch.Tensor):
            tensor = value[0].to(device)
        else:
            tensor = torch.as_tensor(value, device=device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    warn_count = getattr(DPOTrainer, "_ihpn_tensor_patch_warned", 0)
    if warn_count < 5:
        logging.getLogger(__name__).warning(
            "Coerced %s shape %s (type=%s) into tensor for DPO loss",
            name,
            tuple(tensor.shape),
            original_type,
        )
        DPOTrainer._ihpn_tensor_patch_warned = warn_count + 1  # type: ignore[attr-defined]
    return tensor


if not getattr(DPOTrainer, "_ihpn_tensor_patch", False):
    _ORIG_DPO_LOSS = DPOTrainer.dpo_loss

    def _patched_dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        device = getattr(self.accelerator, "device", torch.device("cpu"))
        policy_chosen_logps = _ensure_tensor(
            policy_chosen_logps, device, "policy_chosen_logps"
        )
        policy_rejected_logps = _ensure_tensor(
            policy_rejected_logps, device, "policy_rejected_logps"
        )
        reference_chosen_logps = _ensure_tensor(
            reference_chosen_logps, device, "reference_chosen_logps"
        )
        reference_rejected_logps = _ensure_tensor(
            reference_rejected_logps, device, "reference_rejected_logps"
        )
        return _ORIG_DPO_LOSS(
            self,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

    DPOTrainer.dpo_loss = _patched_dpo_loss  # type: ignore[assignment]
    DPOTrainer._ihpn_tensor_patch = True  # type: ignore[attr-defined]

try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Model registry
MODEL_REGISTRY = {
    "it5": {"prefix": "gsarti/it5-", "type": "t5"},
    "mt5": {"prefix": "google/mt5-", "type": "t5"},
    "flan_t5": {"prefix": "google/flan-t5-", "type": "t5"},
    "bart": {"prefix": "facebook/bart-", "type": "bart"},
}

MODEL_ALIASES = {
    "flan-t5": "flan_t5",
}

MODEL_FAMILY_CHOICES = tuple(
    sorted(set(list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())))
)

VALID_SIZES = ("small", "base", "large")


def _safe_int(val, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float, *, name: str = "value") -> float:
    target = default if val is None else val
    try:
        return float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a float-compatible value, got {val!r}"
        ) from exc


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def extract_prompt_text(example: Mapping[str, Any]) -> str:
    if "prompt" in example and example["prompt"]:
        return str(example["prompt"]).strip()
    instruction = str(example.get("instruction", "") or "")
    input_text = str(example.get("input", "") or "")
    return build_prompt(instruction, input_text).strip()


def extract_reference_text(example: Mapping[str, Any]) -> str:
    for key in ("reference", "output", "chosen", "target"):
        value = example.get(key)
        if value:
            return str(value).strip()
    return ""


class NeutralityClassifier:
    """Runs the hyperpartisan detector on generated text and aggregates metrics."""

    def __init__(
        self,
        model_path: str,
        *,
        neutral_label: str | int = "neutral",
        hyper_label: str | int | None = "hyperpartisan",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        loss_weight: float = 0.0,
        target_rate: Optional[float] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_name)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        label2id = getattr(self.model.config, "label2id", {}) or {}
        self.neutral_id = self._resolve_label_id(neutral_label, label2id)
        self.hyper_id = (
            self._resolve_label_id(hyper_label, label2id)
            if hyper_label is not None
            else None
        )
        self.loss_weight = float(max(0.0, loss_weight))
        if target_rate is not None:
            try:
                self.target_rate = float(target_rate)
            except (TypeError, ValueError):
                self.target_rate = None
        else:
            self.target_rate = None

    @staticmethod
    def _resolve_label_id(label: str | int, mapping: Mapping[str, int]) -> int:
        if isinstance(label, int):
            return label
        if label is None:
            return 0
        if label in mapping:
            return int(mapping[label])
        try:
            return int(label)
        except (TypeError, ValueError):  # pragma: no cover
            raise ValueError(f"Label '{label}' not found in classifier config")

    def _batched(self, texts: Sequence[str]):
        for idx in range(0, len(texts), self.batch_size):
            yield texts[idx : idx + self.batch_size]

    def _score_texts(
        self, texts: Sequence[str]
    ) -> tuple[Dict[str, float | str], torch.Tensor, torch.Tensor, torch.Tensor]:
        if not texts:
            empty_tensor = torch.empty(0, dtype=torch.float32)
            return {}, torch.empty(0, dtype=torch.long), empty_tensor, empty_tensor
        probs_list: List[torch.Tensor] = []
        preds_list: List[torch.Tensor] = []
        with torch.no_grad():
            for chunk in self._batched(texts):
                encoded = self.tokenizer(
                    list(chunk),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                probs_list.append(probs.cpu())
                preds_list.append(preds.cpu())
        probs_tensor = torch.cat(probs_list, dim=0)
        preds_tensor = torch.cat(preds_list, dim=0)
        neutral_probs = probs_tensor[:, self.neutral_id]
        neutral_mask = preds_tensor == self.neutral_id
        neutral_indices = neutral_mask.nonzero(as_tuple=False).view(-1).tolist()
        if self.hyper_id is not None and 0 <= self.hyper_id < probs_tensor.shape[1]:
            hyper_probs = probs_tensor[:, self.hyper_id]
        else:
            hyper_probs = 1.0 - neutral_probs
        neutrality_rate = float((preds_tensor == self.neutral_id).float().mean().item())
        metrics: Dict[str, float | str] = {
            "neutrality_rate": neutrality_rate,
            "hyperpartisan_rate": 1.0 - neutrality_rate,
            "avg_neutrality_prob": float(neutral_probs.mean().item()),
            "avg_hyperpartisan_prob": float(hyper_probs.mean().item()),
            "neutral_prediction_count": float(len(neutral_indices)),
            "neutral_prediction_indexes": json.dumps(neutral_indices),
        }
        if self.loss_weight > 0:
            if self.target_rate is None:
                shortfall = 1.0 - neutrality_rate
            else:
                shortfall = max(0.0, self.target_rate - neutrality_rate)
            metrics["neutrality_loss_term"] = float(self.loss_weight * shortfall)
        return metrics, preds_tensor, neutral_probs, hyper_probs

    def __call__(self, texts: Sequence[str]) -> Dict[str, float | str]:
        metrics, _, _, _ = self._score_texts(texts)
        return metrics

    def detailed_scores(
        self, texts: Sequence[str]
    ) -> tuple[Dict[str, float | str], List[int], List[float], List[float]]:
        metrics, preds_tensor, neutral_probs, hyper_probs = self._score_texts(texts)
        return (
            metrics,
            preds_tensor.tolist(),
            neutral_probs.tolist(),
            hyper_probs.tolist(),
        )


def build_neutrality_classifier(
    config: Mapping[str, Any], logger: logging.Logger
) -> Optional[NeutralityClassifier]:
    if not config:
        return None
    model_path = config.get("model_path")
    if not model_path:
        return None
    try:
        classifier = NeutralityClassifier(
            str(model_path),
            neutral_label=config.get("neutral_label", "neutral"),
            hyper_label=config.get("hyper_label", "hyperpartisan"),
            batch_size=_safe_int(config.get("batch_size", 32), 32),
            max_length=_safe_int(config.get("max_length", 512), 512),
            device=config.get("device"),
            loss_weight=float(config.get("loss_weight", 0.0) or 0.0),
            target_rate=config.get("target_rate"),
        )
        logger.info("Loaded neutrality classifier from %s", model_path)
        return classifier
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize neutrality classifier: %s", exc)
        return None


def _compute_bertscore_lists(
    candidates: Sequence[str], references: Sequence[str], lang: str = "it"
) -> tuple[List[float], List[float], List[float]]:
    if not candidates:
        zeros = [0.0] * len(references)
        return zeros, zeros, zeros
    P, R, F1 = bertscore_score(
        candidates,
        references,
        lang=lang,
        rescale_with_baseline=True,
        verbose=False,
    )
    return (
        P.detach().cpu().tolist(),
        R.detach().cpu().tolist(),
        F1.detach().cpu().tolist(),
    )


def _compute_rouge_l_scores(
    predictions: Sequence[str], references_wrapped: Sequence[Sequence[str]]
) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    values: List[float] = []
    for pred, ref_list in zip(predictions, references_wrapped):
        ref = ref_list[0] if ref_list else ""
        if not pred.strip() or not ref.strip():
            values.append(0.0)
            continue
        values.append(float(scorer.score(ref, pred)["rougeL"].fmeasure))
    avg = float(np.mean(values)) if values else 0.0
    return {"rougeL": avg}


def _compute_bleur_score(corpus_bleu: float, rouge_l: float) -> float:
    bleu_norm = max(0.0, corpus_bleu) / 100.0
    rouge_clamped = max(0.0, min(1.0, rouge_l))
    return float(0.5 * (bleu_norm + rouge_clamped))


def _maybe_load_sentence_transformer(
    model_name: str, logger: logging.Logger
) -> Optional[SentenceTransformer]:
    global _SENTENCE_TRANSFORMER, _SENTENCE_TRANSFORMER_NAME
    if SentenceTransformer is None:
        logger.warning(
            "sentence-transformers not available; skipping Sentence-BERT metrics"
        )
        return None
    if _SENTENCE_TRANSFORMER is None or _SENTENCE_TRANSFORMER_NAME != model_name:
        try:
            _SENTENCE_TRANSFORMER = SentenceTransformer(model_name)
            _SENTENCE_TRANSFORMER_NAME = model_name
            logger.info("Loaded SentenceTransformer model %s", model_name)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to load SentenceTransformer model %s: %s", model_name, exc
            )
            _SENTENCE_TRANSFORMER = None
            _SENTENCE_TRANSFORMER_NAME = None
    return _SENTENCE_TRANSFORMER


def _compute_sbert_similarities(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
    *,
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
    batch_size: int = 32,
) -> Dict[str, Any]:
    if not predictions:
        return {}
    st_model = _maybe_load_sentence_transformer(model_name, logger)
    if st_model is None:
        return {}

    def _encode(texts: Sequence[str]):
        if not texts:
            return None
        return st_model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    limited_sources = sources[: len(predictions)]
    limited_references = references[: len(predictions)]
    source_emb = _encode(limited_sources)
    generated_emb = _encode(predictions)
    reference_emb = _encode(limited_references)
    if source_emb is None or generated_emb is None or reference_emb is None:
        return {}
    source_generated = F.cosine_similarity(source_emb, generated_emb, dim=-1)
    generated_reference = F.cosine_similarity(generated_emb, reference_emb, dim=-1)
    return {
        "source_generated_scores": source_generated.detach().cpu().tolist(),
        "generated_reference_scores": generated_reference.detach().cpu().tolist(),
        "source_generated_avg": float(source_generated.mean().item()),
        "generated_reference_avg": float(generated_reference.mean().item()),
    }


def _maybe_load_comet_model(model_name: str, logger: logging.Logger):
    global _COMET_MODEL, _COMET_MODEL_NAME
    if comet_download_model is None or comet_load_from_checkpoint is None:
        logger.warning("COMET not available; skipping COMET metric")
        return None
    if _COMET_MODEL is None or _COMET_MODEL_NAME != model_name:
        try:
            checkpoint_path = comet_download_model(model_name)
            _COMET_MODEL = comet_load_from_checkpoint(checkpoint_path)
            _COMET_MODEL_NAME = model_name
            logger.info("Loaded COMET model %s", model_name)
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning("Failed to load COMET model %s: %s", model_name, exc)
            _COMET_MODEL = None
            _COMET_MODEL_NAME = None
    return _COMET_MODEL


def _compute_comet_scores(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    logger: logging.Logger,
    *,
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
) -> Dict[str, Any]:
    if not predictions:
        return {}
    comet_model = _maybe_load_comet_model(model_name, logger)
    if comet_model is None:
        return {}
    data = []
    for src, pred, ref in zip(sources, predictions, references):
        data.append({"src": src, "mt": pred, "ref": ref})
    try:
        gpus = 1 if torch.cuda.is_available() else 0
        output = comet_model.predict(data, batch_size=batch_size, gpus=gpus)
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.warning("COMET scoring failed: %s", exc)
        return {}
    system_score = getattr(output, "system_score", None)
    sample_scores = getattr(output, "scores", None)
    if isinstance(output, dict):
        system_score = output.get("system_score", system_score)
        sample_scores = output.get("scores", sample_scores)
    return {
        "system_score": float(system_score) if system_score is not None else 0.0,
        "sample_scores": [float(val) for val in (sample_scores or [])],
    }


def _compute_binary_prf(
    y_true: Sequence[int], y_pred: Sequence[int], positive_label: int
) -> Dict[str, float]:
    if not y_true or not y_pred:
        return {
            "neutrality_precision": 0.0,
            "neutrality_recall": 0.0,
            "neutrality_f1": 0.0,
        }
    limit = min(len(y_true), len(y_pred))
    tp = fp = fn = 0
    for idx in range(limit):
        pred = y_pred[idx]
        truth = y_true[idx]
        if pred == positive_label and truth == positive_label:
            tp += 1
        elif pred == positive_label and truth != positive_label:
            fp += 1
        elif pred != positive_label and truth == positive_label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    return {
        "neutrality_precision": float(precision),
        "neutrality_recall": float(recall),
        "neutrality_f1": float(f1),
    }


@dataclass
class RunConfig:
    model_name: str
    model_type: str
    size: str
    config_path: Path


def load_config(config_path: Path) -> Dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dpo_dataset(train_file: Path, dev_file: Optional[Path] = None) -> DatasetDict:
    """Load DPO preference pairs from JSONL files."""

    def load_jsonl(path: Path) -> List[Dict]:
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    train_data = load_jsonl(train_file)
    logger.info(f"Loaded {len(train_data)} training pairs from {train_file}")

    # Validate data format
    required_keys = {"prompt", "chosen", "rejected"}
    for i, item in enumerate(train_data[:3]):  # Check first 3 items
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"Training data missing required keys: {missing}. "
                f"Item {i}: {list(item.keys())}"
            )
        logger.info(
            f"Sample {i}: prompt_len={len(item['prompt'])}, "
            f"chosen_len={len(item['chosen'])}, "
            f"rejected_len={len(item['rejected'])}"
        )

    # DPO expects 'prompt', 'chosen', 'rejected' columns
    train_dataset = Dataset.from_list(train_data)

    datasets = {"train": train_dataset}

    if dev_file and dev_file.exists():
        dev_data = load_jsonl(dev_file)
        logger.info(f"Loaded {len(dev_data)} dev pairs from {dev_file}")
        datasets["validation"] = Dataset.from_list(dev_data)

    return DatasetDict(datasets)


def _checkpoint_vocab_size(model_dir: Path) -> Optional[int]:
    """Read vocab_size from a saved model directory if available."""

    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read vocab size from %s: %s", config_path, exc)
        return None
    vocab_size = config.get("vocab_size")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    return None


def resolve_best_checkpoint_path(model_dir: Path, logger: logging.Logger) -> Path:
    """Resolve the best checkpoint within an SFT directory, if present."""

    if not model_dir.exists():
        raise FileNotFoundError(f"Provided SFT path does not exist: {model_dir}")

    adapter_file = model_dir / "adapter_config.json"
    model_file = model_dir / "pytorch_model.bin"
    if adapter_file.exists() or model_file.exists():
        return model_dir

    trainer_state = model_dir / "trainer_state.json"
    if trainer_state.exists():
        try:
            with trainer_state.open("r", encoding="utf-8") as fh:
                state = json.load(fh)
            best_checkpoint = state.get("best_model_checkpoint")
            if best_checkpoint:
                best_path = Path(best_checkpoint)
                if not best_path.is_absolute():
                    best_path = model_dir / best_checkpoint
                if best_path.exists():
                    logger.info(
                        "Resolved best checkpoint from trainer_state.json: %s",
                        best_path,
                    )
                    return best_path
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse trainer_state.json: %s", exc)

    checkpoint_dirs = sorted(
        [
            p
            for p in model_dir.iterdir()
            if p.is_dir() and p.name.startswith("checkpoint-")
        ],
        key=lambda path: _safe_int(path.name.split("-")[-1], -1),
    )
    if checkpoint_dirs:
        logger.info(
            "Falling back to latest checkpoint directory: %s", checkpoint_dirs[-1]
        )
        return checkpoint_dirs[-1]

    logger.info("Using provided SFT path as-is (no checkpoints found): %s", model_dir)
    return model_dir


def prepare_model(
    run: RunConfig, gradient_checkpointing: bool = False
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Load tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(run.model_name, use_fast=True)

    # Critical for T5 models: set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For T5, also ensure padding side is correct
    tokenizer.padding_side = "right"  # Important for seq2seq

    model = AutoModelForSeq2SeqLM.from_pretrained(run.model_name)

    # Align embeddings with tokenizer
    model.resize_token_embeddings(len(tokenizer))

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return tokenizer, model


def apply_lora(model, lora_cfg: Dict, model_type: str):
    """Apply LoRA adapters to the model."""
    # Determine target modules based on model type
    if model_type == "bart":
        default_targets = ["q_proj", "v_proj"]
    else:  # t5, mt5
        default_targets = ["q", "v"]

    target_modules = lora_cfg.get("target_modules", default_targets)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.1),
        target_modules=target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


class PerplexityLoggingCallback(TrainerCallback):
    """Log perplexity during training."""

    def on_log(self, args, state: TrainerState, control, logs: Dict, **kwargs):
        if "loss" in logs and logs["loss"] is not None and not math.isnan(logs["loss"]):
            logs["train_perplexity"] = math.exp(min(logs["loss"], 50))
        if (
            "eval_loss" in logs
            and logs["eval_loss"] is not None
            and not math.isnan(logs["eval_loss"])
        ):
            logs["eval_perplexity"] = math.exp(min(logs["eval_loss"], 50))


def evaluate_generation_metrics(
    model,
    tokenizer,
    dataset: Optional[Dataset],
    *,
    model_cfg: Mapping[str, Any],
    generation_cfg: Optional[Mapping[str, Any]],
    neutrality_scorer: Optional[NeutralityClassifier],
    log_dir: Path,
    run_name: str,
    logger: logging.Logger,
) -> Dict[str, float]:
    if dataset is None or len(dataset) == 0:
        logger.warning("No validation set available; skipping seq2seq metrics")
        return {}
    prompts: List[str] = []
    references: List[str] = []
    meta: List[Mapping[str, Any]] = []
    for idx in range(len(dataset)):
        sample = dataset[int(idx)]
        prompt = extract_prompt_text(sample)
        reference = extract_reference_text(sample)
        if not prompt or not reference:
            continue
        prompts.append(prompt)
        references.append(reference)
        meta.append(sample)
    if not prompts:
        logger.warning(
            "Validation set lacks prompt/reference pairs; skipping seq2seq metrics"
        )
        return {}
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dpo_generations_{run_name}.jsonl"
    metrics_path = log_dir / f"dpo_eval_metrics_{run_name}.json"
    gen_kwargs = dict(generation_cfg or {})
    batch_size = max(1, int(gen_kwargs.pop("batch_size", 4) or 4))
    max_input_length = _safe_int(model_cfg.get("max_input_length", 512), 512)
    default_max_length = _safe_int(model_cfg.get("max_target_length", 512), 512)
    gen_kwargs.setdefault("num_beams", 4)
    gen_kwargs.setdefault("max_length", default_max_length)
    predictions: List[str] = []
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            tokenized = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_input_length,
                return_tensors="pt",
            ).to(model_device)
            with torch.no_grad():
                generated_ids = model.generate(**tokenized, **gen_kwargs)
            batch_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            predictions.extend(batch_texts)
    finally:
        if was_training:
            model.train()
    predictions = predictions[: len(references)]
    max_len = len(predictions)
    if max_len == 0:
        logger.warning(
            "Model produced no validation generations; skipping seq2seq metrics"
        )
        return {}
    prompts = prompts[:max_len]
    references = references[:max_len]
    meta = meta[:max_len]
    bleu_metric = load_metric("sacrebleu")
    references_wrapped = [[ref] for ref in references]
    bleu_scores = bleu_metric.compute(
        predictions=predictions, references=references_wrapped
    )
    rouge_scores = _compute_rouge_l_scores(predictions, references_wrapped)
    bert_precisions, bert_recalls, bert_f1s = _compute_bertscore_lists(
        predictions, references
    )
    sbert_metrics = _compute_sbert_similarities(
        prompts, predictions, references, logger
    )
    sbert_source_scores = (
        sbert_metrics.get("source_generated_scores", []) if sbert_metrics else []
    )
    sbert_reference_scores = (
        sbert_metrics.get("generated_reference_scores", []) if sbert_metrics else []
    )
    comet_results = _compute_comet_scores(prompts, predictions, references, logger)
    comet_sentence_scores = (
        comet_results.get("sample_scores", []) if comet_results else []
    )
    pred_lens = [len(pred.split()) for pred in predictions if pred]
    ref_lens = [len(ref.split()) for ref in references if ref]
    metrics: Dict[str, float] = {
        "bleu": float(bleu_scores.get("score", 0.0)),
        "rouge_l": float(rouge_scores.get("rougeL", 0.0)),
        "bertscore_precision": float(
            np.mean(bert_precisions) if bert_precisions else 0.0
        ),
        "bertscore_recall": float(np.mean(bert_recalls) if bert_recalls else 0.0),
        "bertscore_f1": float(np.mean(bert_f1s) if bert_f1s else 0.0),
        "avg_gen_len": float(np.mean(pred_lens) if pred_lens else 0.0),
        "avg_target_len": float(np.mean(ref_lens) if ref_lens else 0.0),
    }
    metrics["bleur"] = _compute_bleur_score(metrics["bleu"], metrics["rouge_l"])
    if metrics["avg_target_len"]:
        metrics["length_ratio"] = metrics["avg_gen_len"] / metrics["avg_target_len"]
    if sbert_metrics:
        metrics["sbert_source_generated"] = sbert_metrics.get(
            "source_generated_avg", 0.0
        )
        metrics["sbert_generated_reference"] = sbert_metrics.get(
            "generated_reference_avg", 0.0
        )
    if comet_results:
        metrics["comet_score"] = comet_results.get("system_score", 0.0)
    neutrality_preds: Optional[List[int]] = None
    neutrality_probs: Optional[List[float]] = None
    neutrality_hyper_probs: Optional[List[float]] = None
    neutrality_truth: Optional[List[int]] = None
    ref_neutral_probs: Optional[List[float]] = None
    ref_hyper_probs: Optional[List[float]] = None
    if neutrality_scorer is not None and predictions:
        (
            neutrality_metrics,
            pred_ids,
            neutral_probs,
            hyper_probs,
        ) = neutrality_scorer.detailed_scores(predictions)
        metrics.update(neutrality_metrics)
        neutrality_preds = [int(pid) for pid in pred_ids]
        neutrality_probs = [float(prob) for prob in neutral_probs]
        neutrality_hyper_probs = [float(prob) for prob in hyper_probs]
        (
            _ref_metrics,
            reference_pred_ids,
            reference_neutral_probs,
            reference_hyper_probs,
        ) = neutrality_scorer.detailed_scores(references)
        neutrality_truth = [int(pid) for pid in reference_pred_ids]
        ref_neutral_probs = [float(prob) for prob in reference_neutral_probs]
        ref_hyper_probs = [float(prob) for prob in reference_hyper_probs]
        metrics["reference_avg_neutrality_prob"] = float(
            np.mean(reference_neutral_probs) if reference_neutral_probs else 0.0
        )
        metrics["reference_avg_hyperpartisan_prob"] = float(
            np.mean(reference_hyper_probs) if reference_hyper_probs else 0.0
        )
        prf_metrics = _compute_binary_prf(
            neutrality_truth, neutrality_preds, neutrality_scorer.neutral_id
        )
        metrics.update(prf_metrics)
    sentence_bleu = sacrebleu.BLEU(smooth_method="exp")
    rouge_metric = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    sentence_bleu_scores: List[float] = []
    with log_path.open("w", encoding="utf-8") as fh:
        for idx, sample in enumerate(meta):
            prompt = prompts[idx]
            reference = references[idx]
            prediction = predictions[idx]
            per_bleu = float(
                sentence_bleu.sentence_score(prediction, [reference]).score
            )
            per_rouge = float(
                rouge_metric.score(reference, prediction)["rougeL"].fmeasure
            )
            sentence_bleu_scores.append(per_bleu)
            record: Dict[str, Any] = {
                "index": idx,
                "prompt": prompt,
                "reference": reference,
                "generated": prediction,
                "bleu_score": per_bleu,
                "rouge_l": per_rouge,
            }
            if idx < len(bert_precisions):
                record["bertscore_precision"] = float(bert_precisions[idx])
            if idx < len(bert_recalls):
                record["bertscore_recall"] = float(bert_recalls[idx])
            if idx < len(bert_f1s):
                record["bertscore_f1"] = float(bert_f1s[idx])
            if idx < len(sbert_source_scores):
                record["sbert_source_generated"] = float(sbert_source_scores[idx])
            if idx < len(sbert_reference_scores):
                record["sbert_generated_reference"] = float(sbert_reference_scores[idx])
            if idx < len(comet_sentence_scores):
                record["comet_score"] = float(comet_sentence_scores[idx])
            if (
                neutrality_preds is not None
                and neutrality_probs is not None
                and idx < len(neutrality_preds)
                and idx < len(neutrality_probs)
            ):
                record["neutrality_prediction"] = neutrality_preds[idx]
                record["neutrality_probability"] = neutrality_probs[idx]
            if neutrality_hyper_probs is not None and idx < len(neutrality_hyper_probs):
                record["neutrality_hyper_probability"] = neutrality_hyper_probs[idx]
            if neutrality_truth is not None and idx < len(neutrality_truth):
                record["neutrality_reference_prediction"] = neutrality_truth[idx]
            if ref_neutral_probs is not None and idx < len(ref_neutral_probs):
                record["neutrality_reference_probability"] = ref_neutral_probs[idx]
            if ref_hyper_probs is not None and idx < len(ref_hyper_probs):
                record["neutrality_reference_hyper_probability"] = ref_hyper_probs[idx]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics["sentence_bleu"] = float(
        np.mean(sentence_bleu_scores) if sentence_bleu_scores else 0.0
    )
    if comet_sentence_scores:
        metrics["comet_sentence_avg"] = float(np.mean(comet_sentence_scores))
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved seq2seq metrics to %s", metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train seq2seq models with DPO")
    parser.add_argument("--train_file", type=Path, default=Path("data/dpo_pairs.jsonl"))
    parser.add_argument("--dev_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/models/dpo"))
    parser.add_argument("--log_dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--model_family", choices=MODEL_FAMILY_CHOICES, default="it5")
    parser.add_argument("--model_size", choices=(*VALID_SIZES, "all"), default="base")
    parser.add_argument("--model_name", type=str, help="Explicit HF model identifier")
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--wandb_project", type=str, default="italian-hyperpartisan-neutralization"
    )
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--sft_model_path", type=Path, help="Path to SFT-finetuned model to start from"
    )
    return parser.parse_args()


def train_dpo(args: argparse.Namespace, run: RunConfig):
    """Run DPO training for a single model configuration."""

    config = load_config(run.config_path) if run.config_path.exists() else {}
    args.log_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    training_cfg = config.get("training", {})
    dpo_cfg = config.get("dpo") or {}
    if not isinstance(dpo_cfg, dict):
        dpo_cfg = {}
    generation_cfg = config.get("generation") or {}
    if not isinstance(generation_cfg, dict):
        generation_cfg = {}
    neutrality_cfg = config.get("neutrality_classifier") or {}
    neutrality_scorer = build_neutrality_classifier(neutrality_cfg, logger)

    # Load dataset
    datasets = load_dpo_dataset(args.train_file, args.dev_file)

    # Load model and tokenizer
    tokenizer, model = prepare_model(run, args.gradient_checkpointing)

    # CRITICAL: Load reference model for DPO - must be loaded separately
    logger.info("Loading reference model for DPO...")
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(run.model_name)
    ref_model.resize_token_embeddings(len(tokenizer))
    # Reference model should be in eval mode
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    sft_checkpoint_path: Optional[Path] = None
    if args.sft_model_path:
        if not args.sft_model_path.exists():
            raise FileNotFoundError(
                f"SFT model path does not exist: {args.sft_model_path}"
            )
        sft_checkpoint_path = resolve_best_checkpoint_path(args.sft_model_path, logger)

    # If starting from SFT checkpoint, load those weights for BOTH models
    if sft_checkpoint_path:
        from peft import PeftModel

        logger.info(f"Loading SFT model from {sft_checkpoint_path}")
        checkpoint_vocab = _checkpoint_vocab_size(sft_checkpoint_path)

        # Load SFT weights into policy model
        if checkpoint_vocab:
            current_vocab = model.get_input_embeddings().weight.shape[0]
            if checkpoint_vocab != current_vocab:
                logger.info(
                    "Resizing embeddings to match SFT checkpoint vocab size (%s -> %s)",
                    current_vocab,
                    checkpoint_vocab,
                )
                model.resize_token_embeddings(checkpoint_vocab)
                ref_model.resize_token_embeddings(checkpoint_vocab)

        model = PeftModel.from_pretrained(model, sft_checkpoint_path)
        model = model.merge_and_unload()  # Merge LoRA weights

        # Also load into reference model
        ref_model = PeftModel.from_pretrained(ref_model, sft_checkpoint_path)
        ref_model = ref_model.merge_and_unload()
        # Keep reference model frozen
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        final_vocab = len(tokenizer)
        merged_vocab = model.get_input_embeddings().weight.shape[0]
        if final_vocab != merged_vocab:
            logger.info(
                "Resizing merged model embeddings to tokenizer vocab size (%s -> %s)",
                merged_vocab,
                final_vocab,
            )
            model.resize_token_embeddings(final_vocab)
            ref_model.resize_token_embeddings(final_vocab)

    # Apply fresh LoRA for DPO (only to policy model, not reference)
    model = apply_lora(model, lora_cfg, run.model_type)

    # Output directory
    safe_model_name = run.model_name.replace("/", "_")
    output_dir = args.output_dir / f"{safe_model_name}_{run.size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # DPO-specific configuration
    beta = _safe_float(dpo_cfg.get("beta"), 0.1, name="dpo.beta")

    # CRITICAL for seq2seq: Disable fp16 if enabled (can cause NaN issues)
    use_fp16 = training_cfg.get("fp16", False)
    use_bf16 = training_cfg.get("bf16", False)

    # If FP16 is causing issues, force it off
    if use_fp16:
        logger.warning(
            "FP16 can cause NaN issues with T5/seq2seq models. Consider using bf16 or fp32."
        )

    # Build DPOConfig - CRITICAL: set is_encoder_decoder=True
    dpo_training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=beta,
        learning_rate=training_cfg.get("learning_rate", 5e-5),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=training_cfg.get("logging_steps", 50),
        eval_strategy=training_cfg.get("evaluation_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 100),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 300),
        load_best_model_at_end=training_cfg.get("load_best_model_at_end", True),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to=["wandb"] if args.use_wandb else [],
        logging_dir=str(args.log_dir),
        # CRITICAL seq2seq parameters
        max_length=model_cfg.get("max_target_length", 512),
        max_prompt_length=model_cfg.get("max_input_length", 512),
        max_target_length=model_cfg.get("max_target_length", 512),
        is_encoder_decoder=True,
        # Important: Don't remove columns - DPO needs them
        remove_unused_columns=False,
        # Important for seq2seq: set label pad token
        label_pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else -100,
    )

    # Initialize wandb
    if args.use_wandb and wandb is not None:
        run_name = (
            args.wandb_run_name or f"dpo-{run.model_name.split('/')[-1]}-{run.size}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "model": run.model_name,
                "size": run.size,
                "beta": beta,
                "is_encoder_decoder": True,
                "config": config,
            },
        )

    # Create DPO trainer - CRITICAL: Pass is_encoder_decoder=True
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        tokenizer=tokenizer,
        callbacks=[PerplexityLoggingCallback()],
    )

    # Train
    logger.info(f"Starting DPO training for {run.model_name}")
    logger.info(f"Model type: seq2seq (encoder-decoder)")
    logger.info(f"Training samples: {len(datasets['train'])}")
    if "validation" in datasets:
        logger.info(f"Validation samples: {len(datasets['validation'])}")

    trainer.train()

    # Save final model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Evaluate
    if datasets.get("validation"):
        eval_results = trainer.evaluate()
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Eval results: {eval_results}")
        seq2seq_metrics = evaluate_generation_metrics(
            model,
            tokenizer,
            datasets["validation"],
            model_cfg=model_cfg,
            generation_cfg=generation_cfg,
            neutrality_scorer=neutrality_scorer,
            log_dir=args.log_dir / "dpo_eval",
            run_name=f"{safe_model_name}_{run.size}",
            logger=logger,
        )
        if seq2seq_metrics:
            with open(
                output_dir / "seq2seq_eval_metrics.json", "w", encoding="utf-8"
            ) as fh:
                json.dump(seq2seq_metrics, fh, indent=2)
            logger.info("Seq2seq metrics: %s", seq2seq_metrics)

    if wandb is not None and wandb.run is not None:
        wandb.finish()

    logger.info(f"DPO training complete. Model saved to {output_dir}")


def main():
    args = parse_args()

    # Canonicalize model family
    if args.model_family in MODEL_ALIASES:
        model_family = MODEL_ALIASES[args.model_family]
    else:
        model_family = args.model_family
    registry_entry = MODEL_REGISTRY.get(model_family)
    if registry_entry is None:
        raise ValueError(f"Unsupported model family: {args.model_family}")

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        prefix = registry_entry.get("prefix")
        if not prefix:
            raise ValueError(f"Missing prefix for model family: {model_family}")
        model_name = f"{prefix}{args.model_size}"

    # Determine model type
    model_type = registry_entry.get("type", "t5")

    # Config path
    config_path = args.config or Path(
        f"experiments/configs/dpo_{model_family}_{args.model_size}.yaml"
    )

    run = RunConfig(
        model_name=model_name,
        model_type=model_type,
        size=args.model_size,
        config_path=config_path,
    )

    train_dpo(args, run)


if __name__ == "__main__":
    main()

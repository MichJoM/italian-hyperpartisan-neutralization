#!/usr/bin/env python
"""Run GPT-4o-mini as an impartial neutrality judge over all texts."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI, OpenAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

JUDGE_PROMPT = """Sei un esperto di analisi del linguaggio giornalistico italiano.
Valuta quanto il seguente testo sia neutrale e imparziale.

Scala di valutazione (1-5):
1 = Molto di parte, linguaggio fortemente emotivo o carico
2 = Parzialmente di parte, diverse espressioni cariche
3 = Moderatamente neutrale, ma alcune tracce di bias
4 = Principalmente neutrale, linguaggio equilibrato
5 = Completamente neutrale, oggettivo e imparziale

Testo da valutare:
{text}

Rispondi SOLO con un JSON in questo formato (nessun testo aggiuntivo):
{{
  "neutrality_score": <1-5>,
  "confidence": <"high"/"medium"/"low">,
  "biased_phrases": [<lista di eventuali frasi di parte trovate>],
  "reasoning": "<una frase che spiega il punteggio>"
}}
"""

INPUT_RATE = 0.00015  # USD per 1K prompt tokens for gpt-4o-mini
OUTPUT_RATE = 0.00060  # USD per 1K completion tokens

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    dataset_path: Path
    metrics_json: Path
    summary_csv: Path
    results_dir: Path
    output_csv: Path
    gpt_summary_csv: Path
    progress_path: Path
    gpt_model: str
    checkpoint_every: int
    resume: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Evaluate neutrality with GPT-4o-mini")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/sft_dev_final.jsonl")
    )
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=Path("outputs/results/automatic_metrics_detailed.json"),
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("outputs/results/automatic_metrics_summary.csv"),
    )
    parser.add_argument("--results_dir", type=Path, default=Path("outputs/results"))
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/results/all_metrics_all_models.csv"),
    )
    parser.add_argument(
        "--gpt_summary_csv",
        type=Path,
        default=Path("outputs/results/gpt4o_mini_LLMs_summary.csv"),
    )
    parser.add_argument(
        "--progress_path",
        type=Path,
        default=Path("outputs/results/gpt4o_judge_progress.jsonl"),
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument(
        "--resume", action="store_true", help="Skip API calls when progress exists"
    )
    args = parser.parse_args()
    return Args(
        dataset_path=args.dataset,
        metrics_json=args.metrics_json,
        summary_csv=args.summary_csv,
        results_dir=args.results_dir,
        output_csv=args.output_csv,
        gpt_summary_csv=args.gpt_summary_csv,
        progress_path=args.progress_path,
        gpt_model=args.gpt_model,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df = df.copy()
    df["dataset_row_id"] = pd.RangeIndex(start=0, stop=len(df))
    df["input_occurrence"] = df.groupby("input").cumcount()
    return df


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Automatic metrics file not found at {path}. Run compute_automatic_metrics.py first."
        )
    return pd.read_json(path)


def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Automatic metrics summary not found at {path}. Run compute_automatic_metrics.py first."
        )
    return pd.read_csv(path)


def md5_hash(text: Any) -> str:
    value = "" if text is None else str(text)
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def build_queue(dataset: pd.DataFrame, metrics: pd.DataFrame) -> List[Dict]:
    queue: List[Dict] = []
    for row in dataset.itertuples():
        queue.append(
            {
                "item_id": f"dataset::original::{row.dataset_row_id}",
                "model_name": "dataset",
                "text_type": "original",
                "entry_id": row.entry_id,
                "paragraph_id": row.paragraph_id,
                "dataset_row_id": row.dataset_row_id,
                "input_occurrence": getattr(row, "input_occurrence", 0),
                "text": row.input,
                "instruction": row.instruction,
                "reference": row.output,
                "input": row.input,
                "text_hash": md5_hash(row.input or ""),
            }
        )
        queue.append(
            {
                "item_id": f"dataset::reference::{row.dataset_row_id}",
                "model_name": "dataset",
                "text_type": "reference",
                "entry_id": row.entry_id,
                "paragraph_id": row.paragraph_id,
                "dataset_row_id": row.dataset_row_id,
                "input_occurrence": getattr(row, "input_occurrence", 0),
                "text": row.output,
                "instruction": row.instruction,
                "reference": row.output,
                "input": row.input,
                "text_hash": md5_hash(row.output or ""),
            }
        )
    for row in metrics.itertuples():
        text = row.generated or ""
        queue.append(
            {
                "item_id": f"{row.model_name}::generated::{row.dataset_row_id}::{row.input_occurrence}",
                "model_name": row.model_name,
                "text_type": "generated",
                "entry_id": row.entry_id,
                "paragraph_id": row.paragraph_id,
                "dataset_row_id": row.dataset_row_id,
                "input_occurrence": row.input_occurrence,
                "text": text,
                "instruction": row.instruction,
                "reference": row.reference,
                "input": row.input,
                "text_hash": md5_hash(text),
            }
        )
    return queue


def load_progress(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    progress: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            progress[record["item_id"]] = record
    LOGGER.info("Loaded %d previously judged samples", len(progress))
    return progress


def append_progress(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class JudgeError(Exception):
    """Wrapper for retryable judge errors."""


def extract_json_payload(text: str) -> Dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise JudgeError("Could not locate JSON object in model response")
    snippet = text[start : end + 1]
    return json.loads(snippet)


def normalize_result(payload: Dict) -> Dict:
    if "neutrality_score" not in payload:
        raise JudgeError("Missing neutrality_score in response")
    score_value = payload["neutrality_score"]
    score = int(score_value)
    confidence = str(payload.get("confidence", "unknown")).lower()
    biased_phrases = payload.get("biased_phrases", [])
    if isinstance(biased_phrases, str):
        biased_phrases = [biased_phrases]
    reasoning = payload.get("reasoning", "").strip()
    return {
        "neutrality_score": score,
        "confidence": confidence,
        "biased_phrases": biased_phrases,
        "reasoning": reasoning,
    }


def requires_api_call(text: str) -> bool:
    return bool(text and text.strip())


def format_biased_phrases(value: Optional[List]) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1000.0) * INPUT_RATE + (
        completion_tokens / 1000.0
    ) * OUTPUT_RATE


def summarise_confidence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for text_type, group in df.groupby("text_type"):
        confidences = group["confidence"].fillna("unknown")
        rows.append(
            {
                "text_type": text_type,
                "mean_score": group["GPT_neutrality_score"].mean(),
                "std_score": group["GPT_neutrality_score"].std(ddof=0),
                "n_high_confidence": int((confidences == "high").sum()),
                "n_medium_confidence": int((confidences == "medium").sum()),
                "n_low_confidence": int((confidences == "low").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set for GPT evaluation.")
    return OpenAI()


def evaluate_queue(
    client: OpenAI,
    queue: List[Dict],
    progress: Dict[str, Dict],
    progress_path: Path,
    model: str,
    checkpoint_every: int,
) -> Dict[str, Dict]:
    prompt_total = 0
    completion_total = 0

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((OpenAIError, JudgeError)),
        reraise=True,
    )
    def call_judge(text: str) -> Dict:
        response = client.responses.create(
            model=model,
            input=JUDGE_PROMPT.format(text=text),
            temperature=0.0,
        )
        content = "".join(block.text for block in response.output[0].content)
        payload = extract_json_payload(content)
        normalized = normalize_result(payload)
        usage = response.usage or {}
        normalized["prompt_tokens"] = usage.get("input_tokens", 0)
        normalized["completion_tokens"] = usage.get("output_tokens", 0)
        return normalized

    processed = 0
    for item in queue:
        item_id = item["item_id"]
        if item_id in progress:
            continue
        text = item["text"] or ""
        if not requires_api_call(text):
            record = {
                **item,
                "GPT_neutrality_score": math.nan,
                "confidence": "unknown",
                "biased_phrases": [],
                "reasoning": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": "empty_text",
            }
        else:
            try:
                result = call_judge(text)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("GPT judge failed for %s", item_id)
                raise exc
            prompt_total += result.pop("prompt_tokens", 0)
            completion_total += result.pop("completion_tokens", 0)
            record = {
                **item,
                "GPT_neutrality_score": result["neutrality_score"],
                "confidence": result["confidence"],
                "biased_phrases": result["biased_phrases"],
                "reasoning": result["reasoning"],
            }
        progress[item_id] = record
        append_progress(progress_path, record)
        processed += 1
        if processed % checkpoint_every == 0:
            LOGGER.info(
                "Processed %d new judgements (%.2f USD so far)",
                processed,
                compute_cost(prompt_total, completion_total),
            )

    if prompt_total or completion_total:
        LOGGER.info(
            "Total GPT tokens — prompt: %d, completion: %d, cost ≈ %.2f USD",
            prompt_total,
            completion_total,
            compute_cost(prompt_total, completion_total),
        )
    return progress


def merge_results(
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    gpt_records: Dict[str, Dict],
    output_csv: Path,
    gpt_summary_csv: Path,
) -> None:
    gpt_df = pd.DataFrame(gpt_records.values())
    if gpt_df.empty:
        raise RuntimeError("No GPT evaluations available — cannot create merged CSV.")

    metrics = metrics.copy()
    metrics["text_type"] = "generated"
    metrics.rename(columns={"generated": "generated_output"}, inplace=True)
    metrics["avg_gen_length_sample"] = metrics["generated_length"]
    summary_subset = summary[
        ["model_name", "classifier_f1", "avg_gen_length", "length_ratio_mean"]
    ].rename(
        columns={
            "classifier_f1": "ft_bert_f1",
            "avg_gen_length": "avg_gen_length_model",
            "length_ratio_mean": "length_ratio_model",
        }
    )
    metrics = metrics.merge(summary_subset, on="model_name", how="left")
    metrics["ft_bert_pred"] = metrics["classifier_predicted_label"]
    metrics["ft_bert_f1"] = metrics["ft_bert_f1"]
    metrics["avg_gen_length"] = metrics.pop("avg_gen_length_model")
    metrics.drop(columns=["length_ratio_model"], inplace=True)

    join_cols = [
        "model_name",
        "text_type",
        "dataset_row_id",
        "paragraph_id",
        "input_occurrence",
    ]
    generated = metrics.merge(gpt_df, on=join_cols, how="left", suffixes=("", "_gpt"))

    generated["biased_phrases"] = generated["biased_phrases"].apply(
        format_biased_phrases
    )
    generated.rename(
        columns={
            "bertscore_precision": "bertscore_precision",
            "bertscore_recall": "bertscore_recall",
            "bertscore_f1": "bertscore_f1",
            "sbert_src_gen": "sbert_src_gen",
            "sbert_gen_ref": "sbert_gen_ref",
            "avg_gen_length_sample": "generated_length",
            "length_ratio": "length_ratio",
            "dataset_row_id": "dataset_row_id",
            "input": "input",
            "reference": "reference",
            "generated_output": "generated_output",
        },
        inplace=True,
    )

    dataset_rows = dataset.copy()
    dataset_rows.rename(columns={"output": "reference"}, inplace=True)
    dataset_rows["model_name"] = "dataset"
    dataset_rows["text_type"] = "original"
    dataset_refs = dataset_rows.copy()
    dataset_refs["text_type"] = "reference"
    dataset_refs["input"] = dataset_refs["reference"]
    dataset_refs["reference"] = dataset_refs["reference"]
    dataset_refs_records = pd.concat([dataset_rows, dataset_refs], ignore_index=True)

    dataset_merge_cols = [
        "model_name",
        "text_type",
        "dataset_row_id",
        "paragraph_id",
        "input_occurrence",
    ]
    dataset_merged = dataset_refs_records.merge(
        gpt_df, on=dataset_merge_cols, how="left"
    )
    dataset_merged["biased_phrases"] = dataset_merged["biased_phrases"].apply(
        format_biased_phrases
    )

    combined = pd.concat([generated, dataset_merged], ignore_index=True, sort=False)
    drop_cols = [
        col
        for col in [
            "text",
            "text_hash",
            "prompt_tokens",
            "completion_tokens",
            "item_id",
        ]
        if col in combined.columns
    ]
    combined.drop(columns=drop_cols, inplace=True, errors="ignore")
    combined.sort_values(
        ["model_name", "text_type", "entry_id", "paragraph_id"], inplace=True
    )
    combined.to_csv(output_csv, index=False)
    LOGGER.info("Saved combined metrics + GPT judgements to %s", output_csv)

    summary_df = summarise_confidence(combined.dropna(subset=["GPT_neutrality_score"]))
    summary_df.to_csv(gpt_summary_csv, index=False)
    LOGGER.info("Saved GPT summary to %s", gpt_summary_csv)


def main() -> None:
    setup_logging()
    args = parse_args()
    ensure_dir(args.results_dir)

    dataset = load_dataset(args.dataset_path)
    metrics = load_metrics(args.metrics_json)
    summary = load_summary(args.summary_csv)

    queue = build_queue(dataset, metrics)
    progress = load_progress(args.progress_path)
    if not args.resume or len(progress) < len(queue):
        client = build_client()
        progress = evaluate_queue(
            client,
            queue,
            progress,
            args.progress_path,
            args.gpt_model,
            args.checkpoint_every,
        )
    else:
        LOGGER.info(
            "Resume flag set and all items already processed; skipping API calls"
        )

    merge_results(
        dataset,
        metrics,
        summary,
        progress,
        args.output_csv,
        args.gpt_summary_csv,
    )


if __name__ == "__main__":
    main()

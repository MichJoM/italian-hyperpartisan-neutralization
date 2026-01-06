"""Prepare SFT and DPO datasets for Italian Hyperpartisan Neutralization."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

INSTRUCTION = (
    "Riscrivi il seguente paragrafo in modo neutrale, rimuovendo ogni tono iper-partitico o di parte, "
    "mantenendo i fatti e il significato originale."
)
SFT_OUTPUT_TRAIN = Path("data/sft_train.jsonl")
SFT_OUTPUT_DEV = Path("data/sft_dev.jsonl")
DPO_OUTPUT = Path("data/dpo_pairs.jsonl")
LOG_PATH = Path("outputs/logs/prepare_datasets.log")

SOURCE_CANDIDATES = ["original_text", "original", "source_text"]
TARGET_PRIORITY = [
    "edited_rewritten_text",
    "rewritten_text",
    "edited_text",
    "neutral_text",
]
LABEL_CANDIDATE = "final_human_label_original"
EDITED_PREFIX = "edited_"


def configure_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("prepare_datasets")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SFT and DPO datasets.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/HIPP_final.csv"),
        help="Input CSV file",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=SFT_OUTPUT_TRAIN,
        help="Output path for SFT train JSONL",
    )
    parser.add_argument(
        "--dev-output",
        type=Path,
        default=SFT_OUTPUT_DEV,
        help="Output path for SFT dev JSONL",
    )
    parser.add_argument(
        "--dpo-output",
        type=Path,
        default=DPO_OUTPUT,
        help="Output path for DPO pairs JSONL",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_PATH,
        help="Path for log file",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty")
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def choose_target_text(row: pd.Series) -> Optional[str]:
    for col in TARGET_PRIORITY:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return None


def choose_source_text(row: pd.Series, source_col: str) -> Optional[str]:
    value = row.get(source_col)
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def prepare_sft_examples(df: pd.DataFrame, source_col: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        source_text = choose_source_text(row, source_col)
        target_text = choose_target_text(row)
        if not source_text or not target_text:
            continue
        records.append(
            {
                "instruction": INSTRUCTION,
                "input": source_text,
                "output": target_text,
            }
        )
    return records


def stratified_split(
    data: List[Dict[str, str]], labels: Optional[pd.Series], train_split: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not data:
        return [], []

    stratify = None
    if labels is not None and labels.nunique(dropna=True) > 1 and len(labels) == len(data):
        stratify = labels

    train, dev = train_test_split(
        data,
        train_size=train_split,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return train, dev


def write_jsonl(path: Path, records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def average_length(records: List[Dict[str, str]], key: str) -> float:
    if not records:
        return 0.0
    lengths = [len(rec[key].split()) for rec in records if rec.get(key)]
    return sum(lengths) / len(lengths)


def prepare_dpo_pairs(df: pd.DataFrame, source_col: str) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    edited_cols = [col for col in df.columns if col.startswith(EDITED_PREFIX)]

    for _, row in df.iterrows():
        source_text = choose_source_text(row, source_col)
        if not source_text:
            continue

        for edited_col in edited_cols:
            edited_text = row.get(edited_col)
            base_col = edited_col.replace(EDITED_PREFIX, "", 1)
            base_text = row.get(base_col)

            if (
                edited_text is None
                or base_text is None
                or pd.isna(edited_text)
                or pd.isna(base_text)
            ):
                continue

            edited_text = str(edited_text).strip()
            base_text = str(base_text).strip()
            if not edited_text or not base_text or edited_text == base_text:
                continue

            prompt = (
                f"{INSTRUCTION}\n\nParagrafo: {source_text}"
            )
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": edited_text,
                    "rejected": base_text,
                    "edited_column": edited_col,
                }
            )
    return pairs


def main() -> None:  # pragma: no cover
    args = parse_args()
    logger = configure_logger(args.log_file)

    try:
        df = load_dataset(args.csv)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    source_col = find_column(df, SOURCE_CANDIDATES)
    if not source_col:
        logger.error("Could not find source column among %s", SOURCE_CANDIDATES)
        sys.exit(1)

    sft_records = prepare_sft_examples(df, source_col)
    label_series = df[LABEL_CANDIDATE] if LABEL_CANDIDATE in df.columns else None
    train_records, dev_records = stratified_split(
        sft_records, label_series, args.train_split, args.seed
    )

    write_jsonl(args.train_output, train_records)
    write_jsonl(args.dev_output, dev_records)

    logger.info(
        "SFT dataset created: %d train, %d dev (avg input len %.1f, avg output len %.1f)",
        len(train_records),
        len(dev_records),
        average_length(sft_records, "input"),
        average_length(sft_records, "output"),
    )

    dpo_pairs = prepare_dpo_pairs(df, source_col)
    write_jsonl(args.dpo_output, dpo_pairs)
    logger.info("DPO pairs created: %d", len(dpo_pairs))


if __name__ == "__main__":
    main()

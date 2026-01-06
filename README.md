# Italian Hyperpartisan Neutralization

Toolkit for building, fine-tuning, and evaluating Italian language models that turn hyperpartisan news paragraphs into neutral rewrites. The repository contains:

- **Dataset preparation** utilities that turn the HIPP corpus into supervised JSONL files and DPO pairs.
- **Supervised fine-tuning (SFT)** pipelines for IT5, FLAN-T5, mT5, and BART models with optional LoRA adapters and neutrality-guided loss.
- **Direct Preference Optimization (DPO)** training with configurable model families, LoRA, and neutrality scoring hooks.
- **Neutrality classifier** training to keep track of hyperpartisan cues and steer generation.
- **Machine text generation** utilities powered by vLLM for running large decoder-only models from templated configs.
- **Dataset documentation** via a full [datasheet](datasheet.md) detailing provenance, licensing, and ethical considerations.
- **Evaluation scripts** ranging from automatic metrics to GPT-4o-mini judging for qualitative inspection.

## Repository Layout

```
├── data/                       # Cannot be disclosed for the moment due to copyrights with Semeval 2023 authors
├── experiments/
│   ├── configs/               # YAML configs for SFT, DPO, and vLLM text generation
│   └── scripts/               # All dataset, training, generation, evaluation, and utility scripts
├── datasheet.md               # Full documentation for the PartisanLens dataset
├── guidelines/                # Project notes and methodology PDFs
├── requirements.txt           # Minimal pip requirements for quick installs
├── t5_environment.yml         # Full Conda environment (CUDA 12 + research stack)
└── outputs/                   # Created automatically by scripts (models, logs, metrics)
```

> **Tip:** Most scripts default to `/italian-hyperpartisan-neutralization/...` paths that were used on the original cluster. Update `BASE_DIR` or wrap scripts via the provided shell launchers before running locally.

## Getting Started

### 1. Prerequisites

- Python 3.10+
- CUDA-ready GPU with ≥16 GB VRAM (24 GB recommended for large IT5 / FLAN-T5 models)
- Git LFS for pulling large checkpoints (if you plan to store them in this repo)
- Optional: [Weights & Biases](https://wandb.ai) account for experiment tracking

### 2. Environment Setup

**Conda (mirrors authors' training env):**

```bash
conda env create -f t5_environment.yml
conda activate t5_env
```

**Pip / virtualenv (leaner stack):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Authenticate with Hugging Face if your base models require it:

```bash
huggingface-cli login
```

Set `WANDB_PROJECT`, `WANDB_ENTITY`, and `WANDB_API_KEY` if you plan to enable W&B logging (`--use_wandb` flag in several scripts).

## Dataset Documentation

The repository ships with [datasheet.md](datasheet.md), a researcher-facing datasheet for the PartisanLens corpus describing motivation, composition, collection protocol, licensing constraints, and recommended/forbidden uses. Please cite or link to the datasheet whenever you redistribute the derived resources, and review it before releasing new annotations to ensure you comply with the underlying SemEval 2023 Task 3 license.


## Supervised Fine-Tuning (SFT)

There are two main SFT entry points under `experiments/scripts/`:

### `sft_it5.py`

Purpose-built for IT5 models with AutoTokenizer, LoRA, gradient checkpointing, and optional neutrality guidance.

1. Edit the **configuration block near the top** to point `BASE_DIR`, `TRAIN_FILE`, `DEV_FILE`, `MODEL_NAME`, and toggles such as `USE_LORA` or `USE_NEUTRALITY_GUIDANCE` to your setup.
2. Launch training:

```bash
python experiments/scripts/sft_it5.py
```

Outputs land in `outputs/models/sft/<model>_lora_*` and include:

- Hugging Face-compatible checkpoints (with LoRA adapters if enabled)
- `trainer_state.json`, metrics, and generated dev predictions for quick sanity checks

### `sft_train_2.py`

More general trainer that works for FLAN-T5, mT5, and BART. It injects the Italian task prefix (`"Riscrivi in modo neutrale:"`) for FLAN-T5, handles tokenizer quirks, and shares the same logging pattern as the IT5 script. Usage mirrors the IT5-only script—edit the constants at the top, then run:

```bash
python experiments/scripts/sft_train_2.py
```

### YAML-Driven SFT

If you prefer declarative configs, use the YAML files in `experiments/configs/` together with the Optuna-ready launchers (e.g., `sft_bart_base_optuna.yaml`). Each config describes:

- `model`: HF checkpoint + sequence lengths
- `lora`: rank, alpha, target modules
- `training`: Trainer arguments, logging cadence, early stopping, and neutrality guidance weight/frequency
- `generation`: Beam search defaults used for validation samples
- `neutrality_classifier`: Path and thresholds for the classifier-assisted loss
- `wandb`: Tags and project metadata

## Neutrality Classifier

`experiments/scripts/corrective_finetune_bert.py` continues fine-tuning an Italian BERT-style classifier on corrective annotations. It supports class-weighted loss, manual evaluation loops, and exports:

- Best checkpoint under `outputs/models/bert_corrected/best`
- Before/after reports (`before_after_comparison.txt`)
- Validation predictions CSVs for auditing

Run it after you have CSVs with `text,label` columns:

```bash
python experiments/scripts/corrective_finetune_bert.py \
  --train_file data/corrective_train.csv \
  --val_file data/corrective_val.csv \
  --base_model dbmdz/bert-base-italian-xxl-uncased \
  --output_dir outputs/models/bert_corrected
```

The resulting classifier path feeds directly into SFT (`NEUTRALITY_CLASSIFIER_PATH`) and DPO configs (`neutrality_classifier.model_path`).

## Direct Preference Optimization (DPO)

`train_dpo_2.py` fine-tunes seq2seq models on preference pairs with TRL's `DPOTrainer` and plugs into the same neutrality scoring stack.

Minimal example:

```bash
python experiments/scripts/train_dpo_2.py \
  --train_file data/dpo_pairs.jsonl \
  --dev_file data/dpo_dev_pairs.jsonl \
  --config experiments/configs/dpo_it5_base.yaml \
  --model_family it5 \
  --model_size base \
  --output_dir outputs/models/dpo \
  --log_dir outputs/logs/dpo \
  --use_wandb \
  --gradient_checkpointing \
  --sft_model_path outputs/models/sft/it5-base_lora_NOCLASS
```

Key features:

- Model registry for IT5, mT5, FLAN-T5, and BART (with automatic tokenizer fixes)
- Optional warm-start from the best SFT checkpoint (`--sft_model_path`)
- Automatic LoRA reapplication for the policy model; frozen reference model
- Built-in BLEU, ROUGE-L, BERTScore, and neutrality metrics on the validation set
- JSONL generation dumps for qualitative inspection (`outputs/models/dpo/.../dpo_generations_*.jsonl`)

## Machine Text Generation (vLLM)

Use `experiments/scripts/generate_machine_text_vllm.py` to batch-generate model completions with [vLLM](https://github.com/vllm-project/vllm) using declarative configs.

1. **Prepare prompts** – The script expects `human_outfox.csv` in the project root with at least `context` and `split` columns. Adjust the path inside the script if your prompt file lives elsewhere.
2. **Pick one or more configs** – Every decoder-only model has a YAML file under `experiments/configs/` (e.g., `gpt-oss-20b.yml`, `gemma-3-12b-it.yml`, `apertus-70b-instruct.yml`, `ministral-3-8b-instruct.yml`). Each config supports:
  - `model_id` or `model_path` (Hugging Face Hub identifier vs local checkpoint path)
  - `init_args` forwarded to `vllm.LLM(...)` (e.g., `tensor_parallel_size`, `dtype`, quantization blocks)
  - `gen_args` consumed by `SamplingParams` (e.g., `max_tokens`, `temperature`, `top_p`)
  - optional `token_args` for tokenizer overrides.
3. **Select the models to run** – Edit the `MODELS` list near the top of the script (or set `MODELS` via your own wrapper) to reference the config basenames you want to execute.
4. **Launch generation** – Run locally or inside SLURM:

```bash
python experiments/scripts/generate_machine_text_vllm.py
```

Set `VERBOSE=1` or `DEBUG=1` for richer logging, and `SAMPLE_PROMPTS=128` to dry-run on a subset. The script detects available GPUs, streams prompts through each model, and saves:

- `outputs/<model>.csv.gz` containing `{prompt, generated_text, split}`
- `outputs/<model>_model_configs.json` documenting the exact config (plus `SLURM_JOB_ID` when present)

VRAM is the limiting factor—refer to the provided configs for tensor parallel settings suitable for H100/L40S clusters.

## Evaluation & Analysis

- **Automatic metrics**: Both SFT and DPO trainers export BLEU/ROUGE/BERTScore summaries to `outputs/models/.../`. Inspect `eval_metrics.json` or the Optuna/W&B dashboards.
- **GPT-4o-mini judge** (`evaluate_gpt4o_judge.py`): Ranks dataset references, model generations, and raw inputs on a 1–5 neutrality scale.
  - Requires `OPENAI_API_KEY` in the environment.
  - Produces merged CSVs (`outputs/results/all_metrics_all_models.csv`) plus per-text JSONL logs and cost tracking.
- **LLM-specific evaluators**: Check `evaluate_llama.py`, `evaluate_gpt4o_judge.py`, and `train_dpo_full_eval.py` for reproducible judge setups.

## Experiment Tracking & Outputs

- **Logs**: Stored under `outputs/logs/` with timestamped files (e.g., dataset prep logs, DPO trainer logs).
- **Models**: Saved under `outputs/models/` (subfolders for `sft`, `dpo`, and `bert_corrected`). LoRA weights and merged checkpoints follow Hugging Face naming conventions.
- **Results**: Automatic metrics, GPT judge summaries, and qualitative samples sit in `outputs/results/`.
- **Scripts**: Shell wrappers (`*.sh`) mirror the Python entry points for batch schedulers—inspect them when porting to Slurm or similar clusters.

## Troubleshooting

- **Tokenizer errors**: Ensure `sentencepiece` is installed (it is in both environment specs) and that you are using the right tokenizer class (IT5 ↔ AutoTokenizer, FLAN-T5 ↔ T5Tokenizer).
- **CUDA OOM**: Lower `per_device_train_batch_size`, reduce `generation_num_beams`, or enable LoRA + gradient checkpointing (all toggles exist in configs).
- **Neutrality classifier missing**: Train it first or set `USE_NEUTRALITY_GUIDANCE = False` / remove `neutrality_classifier` from configs.
- **OpenAI judge throttling**: Use `--checkpoint_every` and `--resume` to recover from interrupted evaluation without repeating expensive API calls.

## Contributing

1. Fork the repository and create a feature branch.
2. Add or update tests/evaluation scripts when changing training logic.
3. Run linters/formatters consistent with `requirements.txt` (HF Transformers, TRL, etc.).
4. Open a pull request that links to reproducible configs or log artifacts.

Feel free to open issues describing new datasets, Italian style guides, or evaluation recipes that would make the neutralization benchmark more robust.

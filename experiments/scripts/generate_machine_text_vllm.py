# vLLM generation script with config loading and multi-model support

import sys
import os
import gc
import random
import warnings
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
from vllm import LLM, SamplingParams

# Control vLLM/engine INFO logging with DEBUG env var
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"
if DEBUG:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Monkeypatch torch.xpu to avoid AttributeError in transformers mxfp4 quantizer
if not hasattr(torch, "xpu"):

    class DummyXPU:
        @staticmethod
        def is_available():
            return False

    torch.xpu = DummyXPU()

# Verbose mode
def parse_bool_env(varname, default="false"):
    val = os.environ.get(varname, default).strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    # fallback: try int conversion, else False
    try:
        return bool(int(val))
    except Exception:
        return False


VERBOSE = parse_bool_env("VERBOSE", "false")

if sys.argv[0]:
    print(f"outputting to: {sys.argv[0]}")
    # exit()
    output_dir = sys.argv[0]
else:
    # Output directory
    output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# Load prompts
df_outfox = pd.read_csv("./human_outfox.csv")
prompts = df_outfox["context"].tolist()
splits = df_outfox["split"].tolist()


# Helper to load model config
def load_model_config(model_name: str) -> dict:
    config_path = Path(f"./configs/{model_name}.yml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



# Harmony prompt formatting for GPT-OSS models
def apply_harmony_format(prompt: str) -> str:
    return (
        # "<|im_start|>system\n"
        # "Reasoning_effort: low\n"
        # "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        # "<|im_start|>assistant\n"
        "<|start|>assistant<|channel|>analysis<|message|><|end|><|start|>assistant\n"
    )


def main():
    # Models to use (edit as needed)
    MODELS = [
        # "mistralai/Mistral-Small-Instruct-2409",
        # "gemma-3-4b-it",
        # "gemma-3-12b-it",
        # "gemma-3-27b-it",
        # "apertus-70b-instruct",
        "gpt-oss-20b",
        # "gpt-oss-120b",
        # "ministral-3-3b-instruct",
        # "ministral-3-8b-instruct",
        # "ministral-3-14b-instruct",
    ]

    # Detect available GPUs 
    num_gpus = torch.cuda.device_count()
    if VERBOSE:
        print(
            f"Detected {num_gpus} CUDA GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}"
        )
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs detected!")

    # Optionally sample prompts for debugging
    val = os.environ.get("SAMPLE_PROMPTS", "")
    SAMPLE_PROMPTS = None if val in ("", "None", "null") else int(val)
    prompts_to_generate = prompts[:SAMPLE_PROMPTS] if SAMPLE_PROMPTS else prompts

    import json
    from datetime import datetime

    for model_name in MODELS:
        print(f"\n{'=' * 50}")
        print(f"Loading config: {model_name}")
        print(f"{'=' * 50}")
        config = load_model_config(model_name)
        # Support both Hugging Face Hub and local models
        model_id = config.get("model_id")
        model_path = config.get("model_path")
        gen_args = config.get("gen_args", {})

        # Pass all gen_args directly to SamplingParams
        sampling_params = SamplingParams(**gen_args, seed=SEED)

        # Apply Harmony format for all gpt-oss models
        if model_name.startswith("gpt-oss"):
            formatted_prompts = [apply_harmony_format(p) for p in prompts_to_generate]
        else:
            formatted_prompts = prompts_to_generate

        # Choose model location: prefer model_id if present, else model_path
        model_location = model_id if model_id else model_path
        if not model_location:
            raise ValueError(
                f"Neither model_id nor model_path specified in config for {model_name}"
            )

        init_args = config.get("init_args", {})
        if VERBOSE:
            print(f"Loading model from: {model_location} with init_args: {init_args}")
        model = LLM(model_location, seed=SEED, **init_args)

        # Generate outputs 
        outputs = model.generate(formatted_prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]

        # Save results
        df_results = pd.DataFrame(
            {
                "prompt": prompts_to_generate,
                "generated_text": generated_texts,
                "split": splits[: len(prompts_to_generate)],
            }
        )
        output_path = output_dir / f"{model_name}.csv.gz"
        df_results.to_csv(output_path, index=False, compression="gzip")
        print(f"Saved outputs to: {output_path}")

        # Save used model config as JSON (for reproducibility)
        config_json = {
            "timestamp": datetime.now().isoformat(),
            "model_configs": {model_name: config},
        }
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id:
            config_json["slurm_job_id"] = slurm_job_id
        config_path = output_dir / f"{model_name}_model_configs.json"
        with open(config_path, "w") as f:
            json.dump(config_json, f, indent=2, default=str)
        if VERBOSE:
            print(f"Saved model config to: {config_path}")

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

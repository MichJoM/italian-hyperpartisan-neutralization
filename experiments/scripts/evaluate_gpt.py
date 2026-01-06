"""
Post-training evaluation script for fine-tuned Llama models
Computes BLEU, ROUGE, BERTScore on the dev set
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from evaluate import load
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================== CONFIGURATION ======================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_WEIGHTS = "./llama3.1-8b-lora-neutralization"  # Path to saved LoRA adapter
DEV_FILE = "/italian-hyperpartisan-neutralization/data/sft_dev.jsonl"
OUTPUT_FILE = "./llama3.1-8b-lora-neutralization/eval_results.json"
HF_TOKEN = os.getenv("HF_TOKEN")

# ====================== LOAD DATA ======================
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

dev_data = load_jsonl(DEV_FILE)
logger.info(f"Loaded {len(dev_data)} dev examples")

# ====================== LOAD MODEL ======================
logger.info(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN,
)

logger.info(f"Loading LoRA weights from: {LORA_WEIGHTS}")
model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
model = model.merge_and_unload()  # Merge LoRA weights for faster inference
model.eval()

# ====================== INFERENCE ======================
predictions = []
references = []

logger.info("Generating predictions...")
for example in tqdm(dev_data):
    messages = [
        {"role": "system", "content": "Sei un assistente esperto nella riscrittura neutrale di testi italiani. Rispondi in modo diretto e oggettivo."},
        {"role": "user", "content": f"{example['instruction']}\n\nParagrafo: {example['input']}"},
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=False,  # Greedy for consistent eval
            pad_token_id=tokenizer.pad_token_id,
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant" in full_output.lower():
        parts = full_output.split("assistant")
        pred = parts[-1].strip()
    else:
        pred = full_output.strip()
    
    predictions.append(pred)
    references.append(example["output"])

# ====================== METRICS ======================
logger.info("Computing metrics...")
bleu = load("bleu")
rouge = load("rouge")
bertscore = load("bertscore")

bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_result = rouge.compute(predictions=predictions, references=references)
bertscore_result = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="it",
    model_type="distilbert-base-multilingual-cased",
)

exact_match = np.mean([
    p.strip().lower() == r.strip().lower() 
    for p, r in zip(predictions, references)
])

results = {
    "bleu": bleu_result["bleu"],
    "rouge1": rouge_result["rouge1"],
    "rouge2": rouge_result["rouge2"],
    "rougeL": rouge_result["rougeL"],
    "bertscore_precision": np.mean(bertscore_result["precision"]),
    "bertscore_recall": np.mean(bertscore_result["recall"]),
    "bertscore_f1": np.mean(bertscore_result["f1"]),
    "exact_match": exact_match,
    "num_examples": len(dev_data),
}

# ====================== SAVE RESULTS ======================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

logger.info(f"Results saved to {OUTPUT_FILE}")
logger.info("\n=== Evaluation Results ===")
for metric, value in results.items():
    logger.info(f"{metric}: {value:.4f}")

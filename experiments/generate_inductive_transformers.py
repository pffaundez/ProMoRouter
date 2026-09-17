#!/usr/bin/env python3
"""Generate the unified inductive test with a sequential Transformers backend.

P0 artifacts are never modified. The script evaluates all 12 models x 6
prompt strategies on the test qids from a persisted split manifest.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


SEEN_MODELS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "yi-34b": "01-ai/Yi-34B-Chat",
    "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
}
SEEN_PROMPTS = ("direct", "cot", "decompose", "selfcheck")
UNSEEN_PROMPTS = ("step_back", "self_consistency")


def normalize(s):
    s = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"[^a-z0-9\s]", "", s)


def extract_final_answer(text):
    """Extract a concise answer for scoring and self-consistency voting."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    answer = lines[-1] if lines else ""
    answer = re.sub(
        r"^(?:the )?(?:final answer|answer|final)\s*(?:(?:is\s*:?)|:)\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    return answer.strip(" \t.\n")


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum(min(p.count(w), g.count(w)) for w in set(g))
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def gsm8k_acc(pred, gold):
    nums = re.findall(r"-?\d+(?:\.\d+)?", (pred or "").replace(",", ""))
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", (gold or "").replace(",", ""))
    return float(bool(nums and gold_nums and nums[-1] == gold_nums[-1]))


def prompt_text(strategy, query, context=""):
    prefix = f"{context}\n\n" if context else ""
    if strategy == "direct":
        return f"Answer the request directly and correctly.\n\n{prefix}{query}"
    if strategy == "cot":
        return f"Think step by step, then provide the final answer.\n\n{prefix}{query}"
    if strategy == "decompose":
        return f"Break the request into short steps, solve it, and provide the final answer.\n\n{prefix}{query}"
    if strategy == "selfcheck":
        return f"Answer, briefly check for errors, then provide the corrected final answer.\n\n{prefix}{query}"
    if strategy == "step_back":
        return f"First identify the general principle needed. Then apply it and provide the final answer.\n\n{prefix}{query}"
    if strategy == "self_consistency":
        return f"Solve independently and provide only the final answer.\n\n{prefix}{query}"
    raise ValueError(strategy)


def load_examples(config_path, needed_tasks):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    all_specs = cfg["tasks"]
    specs = {task: all_specs[task] for task in needed_tasks}
    datasets = {}
    for task, spec in specs.items():
        split = spec.get("split", "train")
        try:
            datasets[task] = load_dataset(
                spec["hf_path"], name=spec.get("hf_name"), split=split
            )
        except ValueError as exc:
            # Cached datasets may expose only a different split (e.g. test).
            if "Unknown split" not in str(exc):
                raise
            datasets[task] = load_dataset(
                spec["hf_path"], name=spec.get("hf_name"), split="test"
            )
    return specs, datasets


def get_example(task, qid, specs, datasets):
    match = re.search(r"(\d+)$", qid)
    if not match:
        raise ValueError(f"Cannot derive dataset index from qid={qid}")
    idx = int(match.group(1))
    row = datasets[task][idx]
    spec = specs[task]
    query = row[spec["query_field"]]
    if spec.get("input_field") and row.get(spec["input_field"]):
        query = f"{query}\n{row[spec['input_field']]}"
    context = row.get(spec.get("context_field", ""), "")
    gold = row[spec["gold_field"]]
    if isinstance(gold, dict):
        gold = (gold.get("text") or [""])[0]
    return str(query), str(context or ""), str(gold)


def generate(model, tokenizer, text, sample=False, seed=0, max_new_tokens=256):
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered = text
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=0.7 if sample else 1.0,
            top_p=0.95,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    generated_tokens = out[0][prompt_len:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer, prompt_len, int(generated_tokens.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-qnorm", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--task-config", type=Path, default=Path("configs/rq2_dataset_builder_smoke.yaml"))
    ap.add_argument("--candidate-manifest", type=Path, default=Path("configs/inductive_candidates.yaml"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--model-ids", nargs="*", default=None)
    ap.add_argument("--prompt-ids", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    qrows = [json.loads(x) for x in args.source_qnorm.read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    test_qids = set(manifest["test_qids"])
    queries = [x for x in qrows if x["qid"] in test_qids]
    if args.max_queries:
        queries = queries[: args.max_queries]

    candidates = yaml.safe_load(args.candidate_manifest.read_text(encoding="utf-8"))
    models = {**SEEN_MODELS, **{x["id"]: x["hf_id"] for x in candidates["unseen_models"]}}
    if args.model_ids:
        models = {k: v for k, v in models.items() if k in set(args.model_ids)}
    prompts = list(SEEN_PROMPTS) + list(UNSEEN_PROMPTS)
    if args.prompt_ids:
        prompts = [p for p in prompts if p in set(args.prompt_ids)]
    print(f"queries={len(queries)} models={len(models)} prompts={len(prompts)} actions={len(queries)*len(models)*len(prompts)}")
    if args.dry_run:
        return

    needed_tasks = {x["task"] for x in queries}
    specs, datasets = load_examples(args.task_config, needed_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for model_id, hf_id in models.items():
            print(f"Loading {model_id} ({hf_id})")
            tokenizer = AutoTokenizer.from_pretrained(hf_id)
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, torch_dtype="auto", device_map="auto"
            )
            model.eval()
            for row in queries:
                query, context, gold = get_example(row["task"], row["qid"], specs, datasets)
                for strategy in prompts:
                    runs = 3 if strategy == "self_consistency" else 1
                    raw_answers, answers, tin, tout, latency = [], [], 0, 0, 0.0
                    for sample_id in range(runs):
                        text = prompt_text(strategy, query, context)
                        t0 = time.time()
                        answer, n_in, n_out = generate(
                            model, tokenizer, text, sample=runs > 1,
                            seed=sample_id, max_new_tokens=args.max_new_tokens
                        )
                        latency += time.time() - t0
                        raw_answers.append(answer)
                        answers.append(extract_final_answer(answer))
                        tin += n_in
                        tout += n_out
                    response = answers[0] if runs == 1 else max(
                        answers, key=lambda x: sum(normalize(x) == normalize(y) for y in answers)
                    )
                    if row["task"] == "gsm8k":
                        performance = gsm8k_acc(response, gold)
                    else:
                        performance = f1(response, gold)
                    record = {
                        "task": row["task"], "qid": row["qid"], "query_text": query,
                        "prompt": strategy, "model": model_id,
                        "hf_id": hf_id, "backend": "transformers",
                        "response": response,
                        "raw_response": raw_answers[0] if runs == 1 else None,
                        "samples": raw_answers if runs > 1 else None,
                        "performance": performance, "gold_source": "huggingface_datasets",
                        "input_tokens": tin, "output_tokens": tout,
                        "tokens_total": tin + tout, "cost_proxy_tokens": tin + tout, "latency_s": latency,
                        "num_samples": runs,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

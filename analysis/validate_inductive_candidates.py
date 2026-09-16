#!/usr/bin/env python3
"""Validate the confirmed inductive candidate manifest and embedding artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml


SEEN_MODELS = {
    "mistral-7b", "qwen2.5-7b", "llama3.1-8b", "qwen2.5-14b",
    "yi-34b", "codellama-34b", "mixtral-8x7b", "llama3.1-70b",
    "qwen2.5-72b",
}
SEEN_PROMPTS = {"direct", "cot", "decompose", "selfcheck"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/inductive_candidates.yaml"))
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/router/inductive_embeddings"))
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    train_models = set(manifest["training_pool"]["models"])
    train_prompts = set(manifest["training_pool"]["prompts"])
    unseen_models = [x["id"] for x in manifest["unseen_models"]]
    unseen_prompts = [x["id"] for x in manifest["unseen_prompts"]]

    errors = []
    if train_models != SEEN_MODELS:
        errors.append(f"training model pool mismatch: {sorted(train_models)}")
    if train_prompts != SEEN_PROMPTS:
        errors.append(f"training prompt pool mismatch: {sorted(train_prompts)}")
    if len(unseen_models) != 3 or len(set(unseen_models)) != 3:
        errors.append("expected exactly 3 unique unseen models")
    if len(unseen_prompts) != 2 or len(set(unseen_prompts)) != 2:
        errors.append("expected exactly 2 unique unseen prompts")
    if train_models & set(unseen_models):
        errors.append("unseen model overlaps training pool")
    if train_prompts & set(unseen_prompts):
        errors.append("unseen prompt overlaps training pool")

    metadata_path = args.embedding_dir / "metadata.json"
    model_path = args.embedding_dir / "unseen_model_embeddings.pt"
    prompt_path = args.embedding_dir / "unseen_prompt_embeddings.pt"
    for path in (metadata_path, model_path, prompt_path):
        if not path.exists():
            errors.append(f"missing artifact: {path}")

    if not errors:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("models") != unseen_models:
            errors.append("metadata model IDs do not match manifest")
        if metadata.get("prompts") != unseen_prompts:
            errors.append("metadata prompt IDs do not match manifest")
        model_pack = torch.load(model_path, map_location="cpu", weights_only=False)
        prompt_pack = torch.load(prompt_path, map_location="cpu", weights_only=False)
        if model_pack["ids"] != unseen_models or tuple(model_pack["embeddings"].shape) != (3, 384):
            errors.append("model embedding pack has wrong IDs or shape")
        if prompt_pack["ids"] != unseen_prompts or tuple(prompt_pack["embeddings"].shape) != (2, 384):
            errors.append("prompt embedding pack has wrong IDs or shape")

    print("==== INDUCTIVE CANDIDATE VALIDATION ====")
    print(f"manifest: {args.manifest}")
    print(f"embedding_dir: {args.embedding_dir}")
    print(f"seen models: {len(train_models)}; unseen models: {len(unseen_models)}")
    print(f"seen prompts: {len(train_prompts)}; unseen prompts: {len(unseen_prompts)}")
    if errors:
        print("validation errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("embedding shapes: models=(3, 384), prompts=(2, 384)")
    print("Validation passed.")


if __name__ == "__main__":
    main()

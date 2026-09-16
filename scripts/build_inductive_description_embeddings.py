#!/usr/bin/env python3
"""Build description embeddings for inductive unseen candidates.

This script intentionally writes only unseen-candidate artifacts. It never
overwrites the closed-pool embeddings under data/router/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def hash_embedding(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
    norm = vec.norm()
    return vec / norm if norm > 0 else vec


def encode(texts, backend, encoder, device, hash_dim):
    if backend == "hash":
        return torch.stack([hash_embedding(t, hash_dim) for t in texts])
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required; install it in the active venv"
        ) from exc
    model = SentenceTransformer(encoder, device=device)
    values = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return values.detach().cpu().float()


def model_text(item):
    return (
        f"Language model: {item['name']}. Family: {item.get('family', 'unknown')}. "
        f"Size: {item['size']}. Description: {item['description']}"
    )


def prompt_text(item):
    return (
        f"Prompting strategy: {item['name']}. Description: {item['description']} "
        f"Expected cost: {item.get('expected_cost', 'unknown')}."
    )


def save_pack(path, ids, texts, matrix):
    torch.save(
        {
            "ids": ids,
            "texts": dict(zip(ids, texts)),
            "embeddings": matrix,
            "embedding_by_id": {k: matrix[i].clone() for i, k in enumerate(ids)},
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("configs/inductive_candidates.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/router/inductive_embeddings"))
    parser.add_argument("--backend", choices=["sentence_transformers", "hash"], default="sentence_transformers")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hash-dim", type=int, default=384)
    args = parser.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    models = manifest["unseen_models"]
    prompts = manifest["unseen_prompts"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_ids = [x["id"] for x in models]
    prompt_ids = [x["id"] for x in prompts]
    model_texts = [model_text(x) for x in models]
    prompt_texts = [prompt_text(x) for x in prompts]

    model_matrix = encode(model_texts, args.backend, args.encoder, args.device, args.hash_dim)
    prompt_matrix = encode(prompt_texts, args.backend, args.encoder, args.device, args.hash_dim)
    save_pack(args.output_dir / "unseen_model_embeddings.pt", model_ids, model_texts, model_matrix)
    save_pack(args.output_dir / "unseen_prompt_embeddings.pt", prompt_ids, prompt_texts, prompt_matrix)

    metadata = {
        "manifest": str(args.manifest),
        "backend": args.backend,
        "encoder": args.encoder if args.backend != "hash" else f"deterministic_hash_{args.hash_dim}",
        "models": model_ids,
        "prompts": prompt_ids,
        "model_shape": list(model_matrix.shape),
        "prompt_shape": list(prompt_matrix.shape),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

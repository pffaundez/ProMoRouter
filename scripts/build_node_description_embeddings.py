#!/usr/bin/env python3
"""
Build text-description embeddings for GraphRouter++ non-query node types:
  - task nodes     from configs/task_descriptions.yaml
  - prompt nodes   from configs/prompt_strategies.yaml
  - model nodes    from configs/model_descriptions.json

Outputs:
  data/router/task_embeddings.pt
  data/router/prompt_embeddings.pt
  data/router/model_embeddings.pt
  data/router/node_description_embedding_metadata.json

Recommended use:
  python scripts/build_node_description_embeddings.py \
    --encoder sentence-transformers/all-MiniLM-L6-v2

The same encoder family should ideally be used for query_embeddings.pt. The trainer uses
separate projection layers, so dimensions may differ, but using a consistent encoder is cleaner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import yaml


DEFAULT_TASK_DESCRIPTIONS = Path("configs/task_descriptions.yaml")
DEFAULT_PROMPT_STRATEGIES = Path("configs/prompt_strategies.yaml")
DEFAULT_MODEL_DESCRIPTIONS = Path("configs/model_descriptions.json")
DEFAULT_OUTPUT_DIR = Path("data/router")


def stable_hash_embedding(text: str, dim: int = 384) -> torch.Tensor:
    """Deterministic fallback for smoke tests only.

    This is not a semantic embedding model. Use --backend sentence_transformers
    for actual experiments.
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = vec.norm(p=2)
    if norm > 0:
        vec = vec / norm
    return vec


class TextEmbedder:
    def __init__(self, backend: str, encoder: str, device: str, hash_dim: int):
        self.backend = backend
        self.encoder = encoder
        self.device = device
        self.hash_dim = hash_dim
        self.model = None

        if backend == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                ) from exc
            self.model = SentenceTransformer(encoder, device=device)
        elif backend == "hash":
            self.model = None
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def encode(self, texts: List[str], batch_size: int) -> torch.Tensor:
        if self.backend == "hash":
            return torch.stack([stable_hash_embedding(t, self.hash_dim) for t in texts], dim=0)

        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return emb.detach().cpu().float()


def load_task_texts(path: Path) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", data)
    out = {}
    for key, item in tasks.items():
        text = f"Task: {item.get('name', key)}. Description: {item['description']}"
        if item.get("primary_metric"):
            text += f" Primary metric: {item['primary_metric']}."
        if item.get("expected_reasoning"):
            text += f" Expected reasoning: {item['expected_reasoning']}."
        out[key] = text
    return out


def load_prompt_texts(path: Path) -> Dict[str, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", data)
    out = {}
    for key, item in prompts.items():
        text = f"Prompting strategy: {item.get('name', key)}. Description: {item['description']}"
        if item.get("expected_cost"):
            text += f" Expected cost: {item['expected_cost']}."
        if item.get("expected_behavior"):
            text += f" Expected behavior: {item['expected_behavior']}."
        out[key] = text
    return out


def load_model_texts(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, item in data.items():
        text = (
            f"Language model: {item.get('name', key)}. "
            f"Family: {item.get('family', 'unknown')}. "
            f"Size: {item.get('size', 'unknown')}. "
            f"Description: {item['description']}"
        )
        out[key] = text
    return out


def save_embedding_pack(name: str, texts_by_id: Dict[str, str], embedder: TextEmbedder, output_dir: Path, batch_size: int):
    ids = list(texts_by_id.keys())
    texts = [texts_by_id[k] for k in ids]
    matrix = embedder.encode(texts, batch_size=batch_size)
    obj = {
        "ids": ids,
        "texts": texts_by_id,
        "embeddings": matrix,
        "embedding_by_id": {k: matrix[i].clone() for i, k in enumerate(ids)},
    }
    out_path = output_dir / f"{name}_embeddings.pt"
    torch.save(obj, out_path)
    print(f"Saved {name} embeddings: {out_path} shape={tuple(matrix.shape)}")
    return out_path, tuple(matrix.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GraphRouter++ node description embeddings.")
    parser.add_argument("--task-descriptions", type=Path, default=DEFAULT_TASK_DESCRIPTIONS)
    parser.add_argument("--prompt-strategies", type=Path, default=DEFAULT_PROMPT_STRATEGIES)
    parser.add_argument("--model-descriptions", type=Path, default=DEFAULT_MODEL_DESCRIPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["sentence_transformers", "hash"], default="sentence_transformers")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hash-dim", type=int, default=384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_texts = load_task_texts(args.task_descriptions)
    prompt_texts = load_prompt_texts(args.prompt_strategies)
    model_texts = load_model_texts(args.model_descriptions)

    embedder = TextEmbedder(args.backend, args.encoder, args.device, args.hash_dim)

    metadata = {
        "backend": args.backend,
        "encoder": args.encoder if args.backend != "hash" else f"deterministic_hash_{args.hash_dim}",
        "files": {},
        "shapes": {},
    }

    for name, texts in [("task", task_texts), ("prompt", prompt_texts), ("model", model_texts)]:
        out_path, shape = save_embedding_pack(name, texts, embedder, args.output_dir, args.batch_size)
        metadata["files"][name] = str(out_path)
        metadata["shapes"][name] = shape

    meta_path = args.output_dir / "node_description_embedding_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()

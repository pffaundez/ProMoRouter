import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

INPUT_PATH = Path("data/interaction_logs/grpp_il_v1/router_model_only_qnorm.jsonl")
OUTPUT_PATH = Path("data/router/query_embeddings.pt")

MODEL_NAME = "intfloat/e5-large-v2"
DEVICE = "cuda:0"


def load_queries():
    queries = []

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]

            query_text = None
            for cand in row["candidates"]:
                query_text = cand.get("query_text")
                if query_text:
                    break

            if query_text is None:
                raise RuntimeError(f"Query text not found for qid={qid}")

            queries.append((qid, query_text))

    return queries


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    queries = load_queries()

    print("==== BUILDING QUERY EMBEDDINGS ====")
    print("Input:", INPUT_PATH)
    print("Output:", OUTPUT_PATH)
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)
    print("Queries:", len(queries))
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(0))

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    texts = [q for _, q in queries]
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    qid_to_emb = {qid: embeddings[i].cpu() for i, (qid, _) in enumerate(queries)}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(qid_to_emb, OUTPUT_PATH)

    print("\nSaved embeddings:")
    print(OUTPUT_PATH)
    print("Embedding dim:", embeddings.shape[1])


if __name__ == "__main__":
    main()
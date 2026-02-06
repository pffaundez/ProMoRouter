from datasets import load_dataset

ds = load_dataset("hotpot_qa", "fullwiki")
ds["train"].to_json("raw/train.jsonl")
ds["validation"].to_json("raw/val.jsonl")
ds["test"].to_json("raw/test.jsonl")
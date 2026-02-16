import argparse
from baselines._common import PROMPTS, chat_completion, accuracy, compute_reward, read_jsonl, write_jsonl, print_summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--prompt", default="direct", choices=list(PROMPTS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    model_pool = [m.strip() for m in args.models.split(",") if m.strip()]
    m = model_pool[-1]
    p = args.prompt

    # same loop as cheapest_fixed
    rows_out = []
    for ex in read_jsonl(args.data):
        sq, gold = ex["subquery"], ex.get("gold", "")
        sys = PROMPTS[p]["system"]
        usr = PROMPTS[p]["user"].format(q=sq)
        rec = dict(task=ex.get("task"), qid=ex.get("qid"), subquery=sq,
                   prompt_id=p, model_id=m, gold=gold, failed=False)
        try:
            res = chat_completion(args.endpoint, m, sys, usr, max_tokens=args.max_tokens)
            rec.update(response=res.text, latency_s=res.latency_s, tokens_in=res.tokens_in, tokens_out=res.tokens_out)
            q = accuracy(res.text, gold) if gold else 0.0
            rec["quality"] = q
            rec["reward"] = compute_reward(q, rec["tokens_in"] + rec["tokens_out"], args.lam)
        except Exception as e:
            rec.update(failed=True, error=str(e), quality=0.0, tokens_in=0, tokens_out=0, latency_s=0.0, reward=0.0)
        rows_out.append(rec)

    write_jsonl(args.out, rows_out)
    print_summary("LargestFixed", rows_out)

if __name__ == "__main__":
    main()

import argparse, random
from baselines._common import PROMPTS, chat_completion, accuracy, compute_reward, read_jsonl, write_jsonl, print_summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with subqueries: {task,qid,subquery,gold}")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--models", required=True, help="Comma-separated model ids (served by endpoint)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    model_pool = [m.strip() for m in args.models.split(",") if m.strip()]
    prompt_pool = list(PROMPTS.keys())

    rows_out = []
    for ex in read_jsonl(args.data):
        sq = ex["subquery"]
        gold = ex.get("gold", "")
        p = random.choice(prompt_pool)
        m = random.choice(model_pool)

        sys = PROMPTS[p]["system"]
        usr = PROMPTS[p]["user"].format(q=sq)

        rec = dict(task=ex.get("task"), qid=ex.get("qid"), subquery=sq,
                   prompt_id=p, model_id=m, gold=gold, failed=False)
        try:
            res = chat_completion(args.endpoint, m, sys, usr, max_tokens=args.max_tokens)
            rec["response"] = res.text
            rec["latency_s"] = res.latency_s
            rec["tokens_in"] = res.tokens_in
            rec["tokens_out"] = res.tokens_out
            q = accuracy(res.text, gold) if gold else 0.0
            rec["quality"] = q
            rec["reward"] = compute_reward(q, rec["tokens_in"] + rec["tokens_out"], args.lam)
        except Exception as e:
            rec["failed"] = True
            rec["error"] = str(e)
            rec["quality"] = 0.0
            rec["tokens_in"] = 0
            rec["tokens_out"] = 0
            rec["latency_s"] = 0.0
            rec["reward"] = 0.0

        rows_out.append(rec)

    write_jsonl(args.out, rows_out)
    print_summary("RandomJoint", rows_out)

if __name__ == "__main__":
    main()

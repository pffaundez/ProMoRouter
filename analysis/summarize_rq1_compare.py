#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_hotpot_rows(rows):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("monolithic") and r.get("decomposed")]
    if not ok:
        return None

    mono_em = mean(r["monolithic"]["em"] for r in ok)
    decomp_em = mean(r["decomposed"]["em"] for r in ok)
    mono_f1 = mean(r["monolithic"]["f1"] for r in ok)
    decomp_f1 = mean(r["decomposed"]["f1"] for r in ok)

    improved = 0
    tied = 0
    worsened = 0
    for r in ok:
        dm = r["decomposed"]["f1"] - r["monolithic"]["f1"]
        if dm > 0:
            improved += 1
        elif dm < 0:
            worsened += 1
        else:
            tied += 1

    return {
        "n_ok": len(ok),
        "monolithic_em": mono_em,
        "decomposed_em": decomp_em,
        "delta_em": decomp_em - mono_em,
        "monolithic_f1": mono_f1,
        "decomposed_f1": decomp_f1,
        "delta_f1": decomp_f1 - mono_f1,
        "improved": improved,
        "tied": tied,
        "worsened": worsened,
    }


def summarize_gsm8k_rows(rows):
    ok = [r for r in rows if r.get("status") == "ok" and r.get("monolithic") and r.get("decomposed")]
    if not ok:
        return None

    mono_em = mean(r["monolithic"]["em"] for r in ok)
    decomp_em = mean(r["decomposed"]["em"] for r in ok)
    mono_tokens = mean(r["monolithic"]["tokens_total"] for r in ok)
    decomp_tokens = mean(r["decomposed"]["tokens_total"] for r in ok)

    improved = 0
    tied = 0
    worsened = 0
    for r in ok:
        dm = r["decomposed"]["em"] - r["monolithic"]["em"]
        if dm > 0:
            improved += 1
        elif dm < 0:
            worsened += 1
        else:
            tied += 1

    return {
        "n_ok": len(ok),
        "monolithic_em": mono_em,
        "decomposed_em": decomp_em,
        "delta_em": decomp_em - mono_em,
        "monolithic_tokens": mono_tokens,
        "decomposed_tokens": decomp_tokens,
        "delta_tokens": decomp_tokens - mono_tokens,
        "tokens_ratio": (decomp_tokens / mono_tokens) if mono_tokens > 0 else None,
        "improved": improved,
        "tied": tied,
        "worsened": worsened,
    }


def print_single_summary(task: str, summary: dict, label: str = None):
    header = task.upper()
    if label:
        header += f" [{label}]"
    print(f"==== {header} SUMMARY ====")
    print(f"N successful: {summary['n_ok']}")

    if task == "hotpotqa":
        print(f"Monolithic EM: {summary['monolithic_em']:.4f}")
        print(f"Decomposed EM: {summary['decomposed_em']:.4f}")
        print(f"Delta EM: {summary['delta_em']:.4f}")
        print(f"Monolithic F1: {summary['monolithic_f1']:.4f}")
        print(f"Decomposed F1: {summary['decomposed_f1']:.4f}")
        print(f"Delta F1: {summary['delta_f1']:.4f}")
        print(f"Improved: {summary['improved']}")
        print(f"Tied: {summary['tied']}")
        print(f"Worsened: {summary['worsened']}")
        print("\nOverleaf row:")
        name = f"{task.capitalize()} (RQ1)" if not label else f"{task.capitalize()} (RQ1, {label})"
        print(
            f"{name} & "
            f"{summary['monolithic_em']:.3f} & "
            f"{summary['decomposed_em']:.3f} & "
            f"{summary['monolithic_f1']:.3f} & "
            f"{summary['decomposed_f1']:.3f} & "
            f"{summary['delta_f1']:.3f} \\\\"
        )

    elif task == "gsm8k":
        print(f"Monolithic EM: {summary['monolithic_em']:.4f}")
        print(f"Decomposed EM: {summary['decomposed_em']:.4f}")
        print(f"Delta EM: {summary['delta_em']:.4f}")
        print(f"Monolithic avg tokens: {summary['monolithic_tokens']:.2f}")
        print(f"Decomposed avg tokens: {summary['decomposed_tokens']:.2f}")
        print(f"Delta avg tokens: {summary['delta_tokens']:.2f}")
        print(f"Token ratio: {summary['tokens_ratio']:.2f}x")
        print(f"Improved: {summary['improved']}")
        print(f"Tied: {summary['tied']}")
        print(f"Worsened: {summary['worsened']}")
        print("\nOverleaf row:")
        name = "GSM8K (RQ1)" if not label else f"GSM8K (RQ1, {label})"
        print(
            f"{name} & "
            f"{summary['monolithic_em']:.3f} & "
            f"{summary['decomposed_em']:.3f} & "
            f"{summary['delta_em']:.3f} & "
            f"{summary['monolithic_tokens']:.1f} & "
            f"{summary['decomposed_tokens']:.1f} & "
            f"{summary['tokens_ratio']:.2f}x \\\\"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")


def print_multi_model_gsm8k(summary_items):
    print("==== GSM8K MULTI-MODEL SUMMARY ====")
    print(
        f"{'tag':<16} {'n_ok':>6} {'mono_em':>10} {'decomp_em':>10} "
        f"{'delta_em':>10} {'mono_tok':>12} {'decomp_tok':>12} {'ratio':>8}"
    )
    print("-" * 90)
    for s in summary_items:
        if s.get("n_ok", 0) == 0:
            print(
                f"{s.get('tag','-'):<16} {0:>6} {'-':>10} {'-':>10} "
                f"{'-':>10} {'-':>12} {'-':>12} {'-':>8}"
            )
            continue
        print(
            f"{s.get('tag','-'):<16} "
            f"{s['n_ok']:>6} "
            f"{s['monolithic_em']:>10.3f} "
            f"{s['decomposed_em']:>10.3f} "
            f"{s['delta_em']:>10.3f} "
            f"{s['monolithic_tokens']:>12.1f} "
            f"{s['decomposed_tokens']:>12.1f} "
            f"{s['tokens_ratio']:>8.2f}x"
        )

    print("\nOverleaf rows:")
    for s in summary_items:
        if s.get("n_ok", 0) == 0:
            continue
        print(
            f"GSM8K (RQ1, {s['tag']}) & "
            f"{s['monolithic_em']:.3f} & "
            f"{s['decomposed_em']:.3f} & "
            f"{s['delta_em']:.3f} & "
            f"{s['monolithic_tokens']:.1f} & "
            f"{s['decomposed_tokens']:.1f} & "
            f"{s['tokens_ratio']:.2f}x \\\\"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["hotpotqa", "gsm8k"])
    ap.add_argument("--input", required=True, help="Path to JSONL result file or multi-model __summary.json")
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".jsonl":
        rows = load_jsonl(p)
        if args.task == "hotpotqa":
            summary = summarize_hotpot_rows(rows)
        elif args.task == "gsm8k":
            summary = summarize_gsm8k_rows(rows)
        else:
            raise ValueError(args.task)

        if summary is None:
            print(f"==== {args.task.upper()} SUMMARY ====")
            print("No successful rows.")
            return

        print_single_summary(args.task, summary)
        return

    if p.suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected summary JSON to contain a list of per-model summaries.")

        if args.task != "gsm8k":
            raise ValueError("Multi-model summary mode is currently supported for gsm8k.")

        print_multi_model_gsm8k(data)
        return

    raise ValueError(f"Unsupported input format: {p.suffix}")


if __name__ == "__main__":
    main()
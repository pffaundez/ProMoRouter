#!/usr/bin/env python3
"""Single held-out-LLM pilot; one GNN, three frozen inference conditions."""
import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits
from router.support_graph import SupportGraphRouter

PROMPTS = ('direct', 'cot', 'decompose', 'selfcheck')
MODELS = ('mistral-7b', 'qwen2.5-7b', 'llama3.1-8b', 'qwen2.5-14b',
          'yi-34b', 'codellama-34b', 'mixtral-8x7b', 'llama3.1-70b', 'qwen2.5-72b')
TASKS = ('gsm8k', 'hotpotqa', 'squad', 'alpaca')


def load_table(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    for key in ('embedding_by_id', 'embeddings_by_id'):
        if isinstance(obj.get(key), dict):
            return obj[key]
    if 'ids' in obj and isinstance(obj.get('embeddings'), torch.Tensor):
        return dict(zip(obj['ids'], obj['embeddings']))
    for key in ('embeddings', 'tensor', 'values'):
        if isinstance(obj.get(key), torch.Tensor):
            return obj[key].float()
    if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f'unsupported embeddings: {path}')


def vector(table, key, order):
    return (table[key] if isinstance(table, dict) else table[order.index(key)]).float().view(-1)


def outcomes(rows, models, seen_normalization=False):
    """Re-normalize seen-only labels without held-out outcomes during fitting."""
    values = []
    for row in rows:
        edges = {(e['prompt'], e['model']): e for e in row['action_edges']}
        data = [[[float(edges[p, m]['performance']), float(edges[p, m]['cost_norm_query'])]
                 for m in models] for p in PROMPTS]
        tensor = torch.tensor(data)
        if seen_normalization:
            cost = tensor[..., 1]
            span = cost.max() - cost.min()
            tensor[..., 1] = (cost - cost.min()) / span if span > 1e-12 else torch.zeros_like(cost)
        values.append(tensor)
    return torch.stack(values)


def select_support(train_ids, k, seed):
    if not 0 < k < len(train_ids):
        raise ValueError('support size must leave nonempty training targets')
    return set(random.Random(seed).sample(sorted(train_ids), k))


def summarize(rows, scores, truth, pool):
    chosen = scores.flatten(1).argmax(-1).tolist()
    flat = truth.flatten(1, 2)
    selected = []
    for i, a in enumerate(chosen):
        p, m = divmod(a, len(pool))
        perf, cost = flat[i, a].tolist()
        selected.append(dict(qid=rows[i]['qid'], task=rows[i]['task'],
                             prompt=PROMPTS[p], model=pool[m], P=perf, C=cost))
    n = len(selected)
    pairs = Counter(f"{x['prompt']}|{x['model']}" for x in selected)
    return dict(queries=n, P=sum(x['P'] for x in selected)/n,
                C=sum(x['C'] for x in selected)/n,
                model_counts=dict(Counter(x['model'] for x in selected)),
                prompt_counts=dict(Counter(x['prompt'] for x in selected)),
                pair_counts=dict(pairs), top_pair_share=max(pairs.values())/n,
                pair_entropy=-sum((v/n)*math.log(v/n) for v in pairs.values()),
                selections=selected)


def add_reward(summary, lam):
    summary['R'] = summary['P'] - lam * summary['C']
    for x in summary['selections']:
        x['R'] = x['P'] - lam*x['C']
    summary['by_task'] = {}
    for task in sorted({x['task'] for x in summary['selections']}):
        selected = [x for x in summary['selections'] if x['task'] == task]
        summary['by_task'][task] = dict(
            queries=len(selected),
            **{k: sum(x[k] for x in selected)/len(selected) for k in ('P', 'C', 'R')},
            pair_counts=dict(Counter(f"{x['prompt']}|{x['model']}" for x in selected)))
    return summary


def incorporation(full, baseline, truth, pool, held, lam):
    reward = truth[..., 0] - lam*truth[..., 1]
    hi = pool.index(held)
    seen_idx = [i for i, m in enumerate(pool) if m != held]
    held_best = reward[:, :, hi].max(1).values
    seen_best = reward[:, :, seen_idx].flatten(1).max(1).values
    opportunities = held_best > seen_best + 1e-7
    selected = [x['model'] == held for x in full['selections']]
    delta = [x['R']-y['R'] for x, y in zip(full['selections'], baseline['selections'])]
    return dict(gain=full['R']-baseline['R'], oracle_gain=float((torch.maximum(held_best, seen_best)-seen_best).mean()),
                oracle_wins=int(opportunities.sum()), held_out_selections=sum(selected),
                model_hits=sum(s and bool(o) for s, o in zip(selected, opportunities)),
                optimal_pairs=sum(s and bool(o) and abs(x['R']-float(b)) < 1e-6
                                  for s, o, x, b in zip(selected, opportunities, full['selections'], held_best)),
                beneficial=sum(d > 1e-7 for s, d in zip(selected, delta) if s),
                harmful=sum(d < -1e-7 for s, d in zip(selected, delta) if s),
                tied=sum(abs(d) <= 1e-7 for s, d in zip(selected, delta) if s))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'query-emb', 'task-emb', 'prompt-emb', 'model-emb'):
        ap.add_argument('--'+name+'-path', type=Path, required=True)
    ap.add_argument('--held-out-model', choices=MODELS, default='qwen2.5-14b')
    ap.add_argument('--calibration-queries', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--reward-weight', type=float, choices=(.1, .5, .9), default=.9)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=18)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--split-manifest', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/support_graph'))
    args = ap.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.hidden_dim) < 1:
        ap.error('epochs/patience/batch-size/hidden-dim must be positive')
    out = args.output_dir / args.held_out_model / f'k{args.calibration_queries}_seed{args.seed}_lam{args.reward_weight}'
    if (out/'results.json').exists() or (out/'router.pt').exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    rows = [json.loads(l) for l in args.data_path.read_text().splitlines() if l.strip()]
    require_complete_action_space(rows, PROMPTS, MODELS)
    if len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('duplicate qids')
    for row in rows:
        for e in row['action_edges']:
            p, c = float(e['performance']), float(e['cost_norm_query'])
            if not math.isfinite(p+c) or not (0 <= p <= 1 and 0 <= c <= 1):
                raise ValueError('invalid performance or cost')
    manifest = args.split_manifest or Path(f'data/router/splits/qnorm_complete_seed{args.seed}.json')
    train, val, test = load_or_create_splits([r['qid'] for r in rows], manifest, args.seed)
    support_ids = select_support(train, args.calibration_queries, args.seed)
    train = train - support_ids
    subset = lambda ids: [r for r in rows if r['qid'] in ids]
    fit, support, validation, targets = map(subset, (train, support_ids, val, test))
    qt, tt, pt, mt = [load_table(p) for p in (args.query_emb_path, args.task_emb_path, args.prompt_emb_path, args.model_emb_path)]
    x = lambda rs: torch.stack([torch.cat((vector(qt, r['qid'], ()), vector(tt, r['task'], TASKS))) for r in rs]).to(device)
    px = torch.stack([vector(pt, p, PROMPTS) for p in PROMPTS]).to(device)
    seen = tuple(m for m in MODELS if m != args.held_out_model)
    mx = lambda pool: torch.stack([vector(mt, m, MODELS) for m in pool]).to(device)
    model = SupportGraphRouter(vector(qt, rows[0]['qid'], ()).numel(), vector(tt, rows[0]['task'], TASKS).numel(), px.shape[1], mx(seen).shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Training and validation context contain seen candidates only.
    support_seen = outcomes(support, seen, True).to(device)
    support_mask = torch.ones(support_seen.shape[:-1], dtype=torch.bool, device=device)
    validation_truth = outcomes(validation, seen, True).to(device)
    best, state, stale, history = -float('inf'), None, 0, []
    for epoch in range(1, args.epochs+1):
        model.train()
        order = list(fit); rng.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start+args.batch_size]
            # All target qids are removed from this batch's support before scoring.
            ids = {r['qid'] for r in batch}
            available = [r for r in fit+support if r['qid'] not in ids]
            context = rng.sample(available, min(args.calibration_queries, len(available)))
            obs = outcomes(context, seen, True).to(device)
            mask = torch.ones(obs.shape[:-1], dtype=torch.bool, device=device)
            scores = model.score(x(batch), model.encode_support(x(context), px, mx(seen), obs, mask))
            y = outcomes(batch, seen, True).to(device)
            reward = y[..., 0]-args.reward_weight*y[..., 1]
            loss = F.mse_loss(scores, reward) + .05*F.cross_entropy(scores.flatten(1), reward.flatten(1).argmax(1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            encoded = model.encode_support(x(support), px, mx(seen), support_seen, support_mask)
            vs = model.score(x(validation), encoded)
            vm = add_reward(summarize(validation, vs.cpu(), validation_truth.cpu(), seen), args.reward_weight)
        history.append(dict(epoch=epoch, loss=sum(losses)/len(losses), validation_R=vm['R']))
        print(f"epoch={epoch} loss={history[-1]['loss']:.5f} val_R={vm['R']:.5f}", flush=True)
        if vm['R'] > best:
            best, stale, best_epoch = vm['R'], 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= args.patience: break
    model.load_state_dict(state); model.eval(); model.requires_grad_(False)
    truth = outcomes(targets, MODELS)
    result = {}
    with torch.no_grad():
        for condition in ('seen_only', 'description_only', 'calibrated'):
            pool = seen if condition == 'seen_only' else MODELS
            obs = outcomes(support, pool).to(device)
            # Normalize support on seen actions only; held-out values cannot
            # change the context of seen candidates in the three conditions.
            si = [pool.index(m) for m in seen]
            c = obs[..., 1]
            lo = c[:, :, si].flatten(1).min(1).values[:, None, None]
            span = c[:, :, si].flatten(1).max(1).values[:, None, None]-lo
            obs[..., 1] = torch.where(span > 1e-12, (c-lo)/span.clamp_min(1e-12), torch.zeros_like(c))
            mask = torch.ones(obs.shape[:-1], dtype=torch.bool, device=device)
            if condition == 'description_only': mask[:, :, pool.index(args.held_out_model)] = False
            encoded = model.encode_support(x(support), px, mx(pool), obs, mask)
            scores = model.score(x(targets), encoded).cpu()
            # Evaluation uses the original fixed full-pool cost reference in
            # every condition. These target costs never enter model inputs.
            observed_truth = truth[:, :, [MODELS.index(m) for m in pool]]
            result[condition] = add_reward(summarize(targets, scores, observed_truth, pool), args.reward_weight)
        for condition in ('description_only', 'calibrated'):
            result[condition]['incorporation'] = incorporation(result[condition], result['seen_only'], truth, MODELS, args.held_out_model, args.reward_weight)
    if any(not torch.equal(value, model.state_dict()[key].cpu()) for key, value in state.items()):
        raise RuntimeError('router weights changed during incorporation')
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = dict(protocol='prompt_conditioned_support_graph_v1', config=config,
                   dataset_sha256=hashlib.sha256(args.data_path.read_bytes()).hexdigest(),
                   split_manifest=str(manifest), train_qids=sorted(train), support_qids=sorted(support_ids),
                   validation_qids=sorted(val), test_qids=sorted(test), seen_models=seen,
                   held_out_calibration_calls=len(support)*len(PROMPTS),
                   support_performance_cost=outcomes(support, MODELS).tolist(),
                   fit_cost_reference='per-query seen-pool minmax',
                   evaluation_cost_reference='stored full-pool qnorm, fixed across conditions',
                   cost_limitation='No net deployment savings claim; calibration token/dollar cost not available here.',
                   best_epoch=best_epoch, history=history, results=result)
    payload['calibration_gain'] = result['calibrated']['R']-result['description_only']['R']
    torch.save(dict(state_dict=state, config=config, support_qids=sorted(support_ids)), out/'router.pt')
    (out/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    for name, metrics in result.items():
        print(name, json.dumps({k: metrics[k] for k in ('P', 'C', 'R', 'top_pair_share')}), metrics.get('incorporation', ''))
    print(f'results={out / "results.json"}')


if __name__ == '__main__':
    main()

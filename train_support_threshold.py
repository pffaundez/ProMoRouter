#!/usr/bin/env python3
"""Bounded nested model-holdout experiment for a frozen-router margin gate."""
import argparse
import hashlib
import json
import math
from collections import Counter
import os
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from train_router_support_graph import (
    MODELS, PROMPTS, TASKS, SupportGraphRouter, add_reward, load_table,
    outcomes, select_support, summarize, vector,
)
from router.data_validation import require_complete_action_space
from router.splits import load_or_create_splits

EXTERNAL = ('yi-34b', 'mixtral-8x7b')
INTERNAL = ('llama3.1-8b', 'codellama-34b')


def known_pool(external=EXTERNAL, held=None):
    return tuple(m for m in MODELS if m not in external and m != held)


def partition_validation(ids, seed):
    ids = sorted(ids)
    random.Random(seed + 2718).shuffle(ids)
    if len(ids) < 4:
        raise ValueError('need at least four validation queries')
    return set(ids[:len(ids)//2]), set(ids[len(ids)//2:])


def policy_records(rows, scores, truth, pool, held, lam):
    """Label the predicted new prompt, never an outcome-selected oracle prompt."""
    hi = pool.index(held)
    si = [i for i, m in enumerate(pool) if m != held]
    base = scores[:, :, si].flatten(1).argmax(1)
    new = scores[:, :, hi].argmax(1)
    if not torch.isfinite(scores).all() or not torch.isfinite(truth).all():
        raise ValueError('nonfinite scores or outcomes')
    records = []
    for i, row in enumerate(rows):
        bp, bm = divmod(int(base[i]), len(si))
        bm = si[bm]
        np = int(new[i])
        p0, c0 = map(float, truth[i, bp, bm])
        p1, c1 = map(float, truth[i, np, hi])
        records.append(dict(qid=row['qid'], task=row['task'], held_out_model=held,
                            margin=float(scores[i, np, hi]-scores[i, bp, bm]),
                            seen_prompt=PROMPTS[bp], seen_model=pool[bm],
                            new_prompt=PROMPTS[np], P_seen=p0, C_seen=c0,
                            P_new=p1, C_new=c1,
                            delta_R=(p1-lam*c1)-(p0-lam*c0)))
    return records


def selected(record, threshold):
    # null denotes reject-all; the strict comparison is part of the protocol.
    return threshold is not None and record['margin'] > threshold


def metrics(records, threshold, lam):
    if not records:
        raise ValueError('empty evaluation')
    picks = [r for r in records if selected(r, threshold)]
    p = sum(r['P_new'] if selected(r, threshold) else r['P_seen'] for r in records)/len(records)
    c = sum(r['C_new'] if selected(r, threshold) else r['C_seen'] for r in records)/len(records)
    pairs = Counter((r['new_prompt'], r['held_out_model']) if selected(r, threshold)
                    else (r['seen_prompt'], r['seen_model']) for r in records)
    n = len(records)
    return dict(P=p, C=c, R=p-lam*c,
                pair_counts={f'{p}|{m}': v for (p, m), v in pairs.items()},
                top_pair_share=max(pairs.values())/n,
                pair_entropy=-sum((v/n)*math.log(v/n) for v in pairs.values()),
                gain=sum(r['delta_R'] for r in picks)/len(records),
                new_selections=len(picks), beneficial=sum(r['delta_R'] > 1e-7 for r in picks),
                harmful=sum(r['delta_R'] < -1e-7 for r in picks),
                tied=sum(abs(r['delta_R']) <= 1e-7 for r in picks))


def choose_threshold(records):
    # These boundary points enumerate every distinct strict-threshold policy.
    margins = sorted({r['margin'] for r in records})
    candidates = [None] + margins + [margins[0]-max(1., abs(margins[0]))]
    def key(t):
        picks = [r for r in records if selected(r, t)]
        return (sum(r['delta_R'] for r in picks)/len(records),
                -len(picks), float('inf') if t is None else t)
    return max(candidates, key=key)


def auc(records):
    # Ignore ties in true reward: they are neither useful nor harmful choices.
    pos = [r['margin'] for r in records if r['delta_R'] > 1e-7]
    neg = [r['margin'] for r in records if r['delta_R'] < -1e-7]
    if not pos or not neg:
        return None
    return sum((p > n)+.5*(p == n) for p in pos for n in neg)/(len(pos)*len(neg))


def internal_gate(records, threshold, lam):
    folds = {m: metrics([r for r in records if r['held_out_model'] == m], threshold, lam)
             for m in sorted({r['held_out_model'] for r in records})}
    pooled = metrics(records, threshold, lam)
    fold_auc = {m: auc([r for r in records if r['held_out_model'] == m]) for m in folds}
    a = auc(records)
    passed = (threshold is not None and a is not None and a > .5
              and pooled['gain'] > 0 and all(v['gain'] >= 0 for v in folds.values())
              and all(v is not None and v > .5 for v in fold_auc.values()))
    return dict(passed=passed, pooled_auc=a, by_model_auc=fold_auc, pooled=pooled, by_model=folds,
                rule='pooled and each fold AUC > 0.5, pooled gain > 0, every fold gain >= 0; reject-all fails')


def cluster_interval(records, threshold, seed):
    groups = {}
    for r in records:
        groups.setdefault(r['qid'], []).append(r['delta_R'] if selected(r, threshold) else 0.)
    values = list(groups.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        sample = [rng.choice(values) for _ in values]
        draws.append(sum(map(sum, sample))/sum(map(len, sample)))
    draws.sort()
    return [draws[49], draws[1949]]


class Experiment:
    def __init__(self, args, rows, support, fit):
        self.args, self.rows, self.support, self.fit = args, rows, support, fit
        self.device = torch.device(args.device)
        qt, tt, pt, mt = [load_table(p) for p in (args.query_emb_path, args.task_emb_path,
                                                args.prompt_emb_path, args.model_emb_path)]
        self.qdim = vector(qt, rows[0]['qid'], ()).numel()
        self.tdim = vector(tt, rows[0]['task'], TASKS).numel()
        self.x = lambda rs: torch.stack([torch.cat((vector(qt, r['qid'], ()), vector(tt, r['task'], TASKS))) for r in rs]).to(self.device)
        self.px = torch.stack([vector(pt, p, PROMPTS) for p in PROMPTS]).to(self.device)
        self.mx = lambda pool: torch.stack([vector(mt, m, MODELS) for m in pool]).to(self.device)

    def fit_router(self, seen, validation, name):
        a = self.args
        torch.manual_seed(a.seed)
        rng = random.Random(a.seed)
        net = SupportGraphRouter(self.qdim, self.tdim, self.px.shape[1],
                                 self.mx(seen).shape[1], 128, 'uniform').to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
        obs = outcomes(self.support, seen, True).to(self.device)
        mask = torch.ones(obs.shape[:-1], dtype=torch.bool, device=self.device)
        truth = outcomes(validation, seen, True)
        best, state, stale, history = -float('inf'), None, 0, []
        for epoch in range(1, a.epochs+1):
            net.train()
            order = list(self.fit); rng.shuffle(order)
            losses = []
            for start in range(0, len(order), 32):
                batch = order[start:start+32]
                ids = {r['qid'] for r in batch}
                available = [r for r in self.fit+self.support if r['qid'] not in ids]
                context = rng.sample(available, min(8, len(available)))
                co = outcomes(context, seen, True).to(self.device)
                cm = torch.ones(co.shape[:-1], dtype=torch.bool, device=self.device)
                scores = net.score(self.x(batch), net.encode_support(self.x(context), self.px, self.mx(seen), co, cm))
                y = outcomes(batch, seen, True).to(self.device)
                reward = y[..., 0]-.9*y[..., 1]
                loss = F.mse_loss(scores, reward)+.05*F.cross_entropy(scores.flatten(1), reward.flatten(1).argmax(1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step()
                losses.append(float(loss.detach()))
            net.eval()
            with torch.no_grad():
                scores = net.score(self.x(validation), net.encode_support(self.x(self.support), self.px, self.mx(seen), obs, mask)).cpu()
            reward = add_reward(summarize(validation, scores, truth, seen), .9)['R']
            history.append(dict(epoch=epoch, loss=sum(losses)/len(losses), validation_R=reward))
            print(f'{name} epoch={epoch} val_R={reward:.6f}', flush=True)
            if reward > best:
                best, stale, best_epoch = reward, 0, epoch
                state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= a.patience:
                break
        net.load_state_dict(state); net.eval(); net.requires_grad_(False)
        torch.save(dict(state_dict=state, seen_models=seen, best_epoch=best_epoch,
                        history=history), a.output_dir/f'{name}.pt')
        return net, dict(seen_models=seen, best_epoch=best_epoch, history=history)

    def evaluate(self, net, seen, held, targets):
        pool = tuple(m for m in MODELS if m in seen or m == held)
        obs = outcomes(self.support, pool).to(self.device)
        si = [pool.index(m) for m in seen]
        c = obs[..., 1]
        lo = c[:, :, si].flatten(1).min(1).values[:, None, None]
        span = c[:, :, si].flatten(1).max(1).values[:, None, None]-lo
        obs[..., 1] = torch.where(span > 1e-12, (c-lo)/span.clamp_min(1e-12), torch.zeros_like(c))
        mask = torch.ones(obs.shape[:-1], dtype=torch.bool, device=self.device)
        before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        with torch.no_grad():
            scores = net.score(self.x(targets), net.encode_support(self.x(self.support), self.px, self.mx(pool), obs, mask)).cpu()
            base_obs = outcomes(self.support, seen, True).to(self.device)
            base_mask = torch.ones(base_obs.shape[:-1], dtype=torch.bool, device=self.device)
            base_scores = net.score(self.x(targets), net.encode_support(self.x(self.support), self.px, self.mx(seen), base_obs, base_mask)).cpu()
        if not torch.allclose(scores[:, :, si], base_scores, atol=1e-6, rtol=1e-6):
            raise RuntimeError('adding candidate changed seen scores')
        if any(not torch.equal(v, net.state_dict()[k]) for k, v in before.items()):
            raise RuntimeError('inference changed router state')
        return policy_records(targets, scores, outcomes(targets, pool), pool, held, .9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'query-emb', 'task-emb', 'prompt-emb', 'model-emb'):
        ap.add_argument('--'+name+'-path', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--patience', type=int, default=18)
    ap.add_argument('--split-manifest', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/support_threshold/seed1_lam0.9'))
    a = ap.parse_args()
    if min(a.epochs, a.patience) < 1:
        ap.error('epochs and patience must be positive')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    rows = [json.loads(l) for l in a.data_path.read_text().splitlines() if l.strip()]
    require_complete_action_space(rows, PROMPTS, MODELS)
    if len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('duplicate qids')
    manifest = a.split_manifest or Path(f'data/router/splits/qnorm_complete_seed{a.seed}.json')
    train, val, test = load_or_create_splits([r['qid'] for r in rows], manifest, a.seed)
    support_ids = select_support(train, 8, a.seed)
    tune_ids, check_ids = partition_validation(val, a.seed)
    subset = lambda ids: [r for r in rows if r['qid'] in ids]
    support, fit, tune, check, validation, targets = map(subset, (support_ids, train-support_ids, tune_ids, check_ids, val, test))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    result = dict(protocol='support_margin_gate_v1', config=config,
                  dataset_sha256=hashlib.sha256(a.data_path.read_bytes()).hexdigest(),
                  external_models=EXTERNAL, internal_models=INTERNAL,
                  train_qids=sorted(train-support_ids), support_qids=sorted(support_ids),
                  tune_qids=sorted(tune_ids), check_qids=sorted(check_ids), test_qids=sorted(test),
                  reward='P - 0.9*C; evaluation C = original full-pool qnorm',
                  support_calls_per_new_model=32, fitting_cost='per-query seen-only minmax',
                  internal_checkpoints={}, internal_records={})
    def save():
        (a.output_dir/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    result['status'] = 'running_internal'; save()
    exp = Experiment(a, rows, support, fit)
    tune_records, check_records = [], []
    for held in INTERNAL:
        seen = known_pool(held=held)
        net, history = exp.fit_router(seen, tune, 'internal_'+held)
        result['internal_checkpoints'][held] = history
        tr, cr = exp.evaluate(net, seen, held, tune), exp.evaluate(net, seen, held, check)
        tune_records.extend(tr); check_records.extend(cr)
        del net
    threshold = choose_threshold(tune_records)
    gate = internal_gate(check_records, threshold, .9)
    gate['gain_cluster_bootstrap_95ci'] = cluster_interval(check_records, threshold, a.seed)
    result.update(threshold=threshold, internal_gate=gate,
                  internal_records=dict(tune=tune_records, check=check_records))
    frozen = dict(threshold=threshold, comparison='margin > threshold; null = reject-all',
                  internal_gate=gate, external_models=EXTERNAL,
                  dataset_sha256=result['dataset_sha256'], tune_qids=sorted(tune_ids),
                  check_qids=sorted(check_ids), config=config)
    # Persist the decision BEFORE any external calibration outcomes or test labels.
    freeze_path = a.output_dir/'frozen_threshold.json'
    freeze_path.write_text(json.dumps(frozen, indent=2, allow_nan=False)+'\n')
    freeze_hash = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    result['frozen_threshold_sha256'] = freeze_hash
    if not gate['passed']:
        result['status'] = 'stopped_internal_gate'; save()
        print('STOP: internal gate failed; external models were not evaluated.', flush=True)
    else:
        result['status'] = 'running_external'; save()
        seen = known_pool()
        net, history = exp.fit_router(seen, validation, 'external_shared')
        result['external_checkpoint'] = history
        result['external_results'] = {}
        for held in EXTERNAL:
            records = exp.evaluate(net, seen, held, targets)
            result['external_results'][held] = dict(
                records=records, reject_all=metrics(records, None, .9),
                original_argmax=metrics(records, 0., .9), gated=metrics(records, threshold, .9),
                gain_bootstrap_95ci=cluster_interval(records, threshold, a.seed),
                margin_auc=auc(records),
                by_task={task: {name: metrics([r for r in records if r['task'] == task], t, .9)
                                for name, t in (('reject_all', None), ('original_argmax', 0.), ('gated', threshold))}
                         for task in sorted({r['task'] for r in records})})
        if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != freeze_hash:
            raise RuntimeError('frozen threshold changed during confirmation')
        result['external_confirmation_passed'] = all(
            v['gated']['gain'] > 0 and v['gated']['R'] > v['original_argmax']['R']
            for v in result['external_results'].values())
        result['status'] = 'completed_external'; save()
        print(json.dumps({m: v['gated'] for m, v in result['external_results'].items()}, indent=2))
    print(f'results={a.output_dir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()

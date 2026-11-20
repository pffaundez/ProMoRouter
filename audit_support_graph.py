#!/usr/bin/env python3
"""Audit an existing support-graph checkpoint without training or selection on test."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

from router.support_graph import SupportGraphRouter
from train_router_support_graph import (
    MODELS, PROMPTS, TASKS, add_reward, load_table, outcomes, summarize, vector,
)


def score_diagnostics(before, after, truth, held_index, lam, qids):
    """Positive margin means the best held-out score beats the best seen score."""
    before, after, truth = before.double(), after.double(), truth.double()
    seen = [i for i in range(after.shape[2]) if i != held_index]
    reward = truth[..., 0] - lam * truth[..., 1]
    opportunities = reward[:, :, held_index].max(1).values > reward[:, :, seen].flatten(1).max(1).values + 1e-7
    records = []
    for i, qid in enumerate(qids):
        rec = dict(qid=qid, oracle_opportunity=bool(opportunities[i]), prompts={})
        for label, scores in (('before', before), ('after', after)):
            h = scores[i, :, held_index]
            seen_best = scores[i, :, seen].max().item()
            best = h.max().item()
            rec[label] = dict(best_score=best, best_prompt=PROMPTS[int(h.argmax())],
                              best_pair_rank=1+int((scores[i] > best).sum()),
                              margin_to_seen=best-seen_best)
        for p, prompt in enumerate(PROMPTS):
            a, b = before[i, p, held_index].item(), after[i, p, held_index].item()
            rec['prompts'][prompt] = dict(before=a, after=b, delta=b-a)
        rec['best_score_delta'] = rec['after']['best_score']-rec['before']['best_score']
        records.append(rec)
    groups = {}
    for name, selected in (('all_queries', records),
                           ('oracle_opportunities', [r for r in records if r['oracle_opportunity']])):
        if not selected:
            groups[name] = dict(queries=0)
            continue
        item = dict(queries=len(selected))
        for label in ('before', 'after'):
            item[label] = dict(
                median_best_pair_rank=statistics.median(r[label]['best_pair_rank'] for r in selected),
                mean_margin_to_seen=statistics.mean(r[label]['margin_to_seen'] for r in selected),
                median_margin_to_seen=statistics.median(r[label]['margin_to_seen'] for r in selected),
                strictly_beats_seen=sum(r[label]['margin_to_seen'] > 0 for r in selected),
                best_prompt_counts={p: sum(r[label]['best_prompt']==p for r in selected) for p in PROMPTS})
        item['mean_best_score_delta'] = statistics.mean(r['best_score_delta'] for r in selected)
        item['per_prompt'] = {p: dict(
            mean_delta=statistics.mean(r['prompts'][p]['delta'] for r in selected),
            mean_abs_delta=statistics.mean(abs(r['prompts'][p]['delta']) for r in selected),
            max_abs_delta=max(abs(r['prompts'][p]['delta']) for r in selected)) for p in PROMPTS}
        groups[name] = item
    return dict(summary=groups, per_query=records,
                max_abs_held_score_delta=float((after[:, :, held_index]-before[:, :, held_index]).abs().max()),
                max_abs_seen_score_delta=float((after[:, :, seen]-before[:, :, seen]).abs().max()))


def fixed_task_actions(validation, truth, pool, lam):
    """Choose among seen pairs using validation outcomes only, never test."""
    reward = truth[..., 0]-lam*truth[..., 1]
    choices = {}
    global_best = int(reward.flatten(1).mean(0).argmax())
    for task in TASKS:
        indices = [i for i, r in enumerate(validation) if r['task']==task]
        a = int(reward[indices].flatten(1).mean(0).argmax()) if indices else global_best
        choices[task] = dict(prompt=PROMPTS[a//len(pool)], model=pool[a%len(pool)],
                             action_index=a, validation_queries=len(indices),
                             fallback_global=not bool(indices))
    return choices


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    source = args.run_dir/'results.json'
    original = json.loads(source.read_text())
    cfg = original['config']
    destination = args.json_out or args.run_dir/'score_audit.json'
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite {destination}')
    raw = Path(cfg['data_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != original['dataset_sha256']:
        raise ValueError('dataset differs from the training run')
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    by_id = {r['qid']: r for r in rows}
    if len(by_id)!=len(rows): raise ValueError('duplicate dataset qids')
    sets = [set(original[k]) for k in ('train_qids','support_qids','validation_qids','test_qids')]
    if sum(map(len, sets)) != len(set.union(*sets)) or set.union(*sets)!=set(by_id):
        raise ValueError('invalid partitions')
    # Preserve the dataset order used by the trainer.
    support, validation, targets = ([r for r in rows if r['qid'] in ids] for ids in sets[1:])
    seen = tuple(original['seen_models'])
    held = cfg['held_out_model']; hi = MODELS.index(held); lam = cfg['reward_weight']
    device = torch.device(args.device)
    qt, tt, pt, mt = [load_table(Path(cfg[name+'_path'])) for name in ('query_emb','task_emb','prompt_emb','model_emb')]
    x = lambda rs: torch.stack([torch.cat((vector(qt,r['qid'],()),vector(tt,r['task'],TASKS))) for r in rs]).to(device)
    px = torch.stack([vector(pt,p,PROMPTS) for p in PROMPTS]).to(device)
    mx = torch.stack([vector(mt,m,MODELS) for m in MODELS]).to(device)
    model = SupportGraphRouter(vector(qt,rows[0]['qid'],()).numel(),vector(tt,rows[0]['task'],TASKS).numel(),px.shape[1],mx.shape[1],cfg['hidden_dim'],cfg.get('support_weighting','uniform')).to(device)
    checkpoint = torch.load(args.run_dir/'router.pt',map_location='cpu',weights_only=True)
    if checkpoint['config']!=cfg or sorted(checkpoint['support_qids'])!=sorted(original['support_qids']):
        raise ValueError('checkpoint/results identity mismatch')
    model.load_state_dict(checkpoint['state_dict']);model.eval();model.requires_grad_(False)
    support_truth = outcomes(support,MODELS)
    if not torch.allclose(support_truth,torch.tensor(original['support_performance_cost']),atol=1e-7,rtol=0):
        raise ValueError('support outcomes differ from stored run')
    obs = support_truth.to(device)
    si = [MODELS.index(m) for m in seen]
    c = obs[...,1]
    lo = c[:,:,si].flatten(1).min(1).values[:,None,None]
    span = c[:,:,si].flatten(1).max(1).values[:,None,None]-lo
    obs[...,1]=torch.where(span>1e-12,(c-lo)/span.clamp_min(1e-12),torch.zeros_like(c))
    mask = torch.ones(obs.shape[:-1],dtype=torch.bool,device=device)
    scores={}
    with torch.inference_mode():
        for condition in ('description_only','calibrated'):
            visible=mask.clone()
            if condition=='description_only': visible[:,:,hi]=False
            encoded=model.encode_support(x(support),px,mx,obs,visible,target_x=x(targets))
            scores[condition]=model.score(x(targets),encoded).cpu()
    truth=outcomes(targets,MODELS)
    replay={}
    for condition,sc in scores.items():
        replay[condition]=add_reward(summarize(targets,sc,truth,MODELS),lam)
        expected=original['results'][condition]
        if [(r['qid'],r['prompt'],r['model']) for r in replay[condition]['selections']] != [(r['qid'],r['prompt'],r['model']) for r in expected['selections']]:
            raise ValueError(f'{condition}: selections do not reproduce; rerun audit on the original device')
        if any(abs(replay[condition][k]-expected[k])>1e-6 for k in ('P','C','R')):
            raise ValueError('replayed metrics differ from run')
    diag=score_diagnostics(scores['description_only'],scores['calibrated'],truth,hi,lam,[r['qid'] for r in targets])
    # Match the seen-only validation cost scale used to select the checkpoint.
    choices=fixed_task_actions(validation,outcomes(validation,seen,True),seen,lam)
    synthetic_scores=torch.full((len(targets),len(PROMPTS),len(seen)),-1.)
    for i,row in enumerate(targets):
        a=choices[row['task']]['action_index']
        synthetic_scores[i].flatten()[a]=1.
    fixed=add_reward(summarize(targets,synthetic_scores,truth[:,:,si],seen),lam)
    payload=dict(protocol='support_graph_score_audit_v1',source_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 checkpoint_sha256=hashlib.sha256((args.run_dir/'router.pt').read_bytes()).hexdigest(),
                 device=args.device,held_out_model=held,lambda_value=lam,reproduced_selections=True,
                 score_diagnostics=diag,fixed_pair_by_task=dict(
                     selection_reference='seen-only validation, seen-normalized cost; no test selection',
                     choices=choices,test_metrics=fixed,
                     router_minus_fixed_R=original['results']['calibrated']['R']-fixed['R']),
                 non_claim='Audit only; no retraining, tuning on test, or causal attribution from score change alone.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(diag['summary'],indent=2))
    print(f"fixed_by_task R={fixed['R']:.6f} P={fixed['P']:.6f} C={fixed['C']:.6f} router_minus_fixed={payload['fixed_pair_by_task']['router_minus_fixed_R']:+.6f}")
    print(f'audit={destination}')


if __name__=='__main__': main()

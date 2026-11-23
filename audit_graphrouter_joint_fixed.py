#!/usr/bin/env python3
"""Final fixed-policy audit: standard library only, no inference or training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def reward(row, action, lo, span, lam):
    edges = [e for e in row['action_edges'] if (e['prompt'],e['model']) == action]
    if len(edges) != 1:
        raise ValueError(f'missing/duplicated action {row["qid"]} {action}')
    p, money = (float(edges[0][k]) for k in ('performance','cost_proxy_money'))
    if not all(math.isfinite(x) for x in (p,money)) or not (0 <= p <= 1 and money >= 0):
        raise ValueError('invalid permitted outcome')
    return p-lam*(money-lo)/span


def choose_policies(validation, support, prompts, seen, held, lo, span, lam):
    """This selector cannot read test or unseen validation outcomes."""
    average = lambda rows,a: sum(reward(r,a,lo,span,lam) for r in rows)/len(rows)
    actions = [(p,m) for p in prompts for m in seen]
    known = max(actions,key=lambda a:average(validation,a))
    new = max([(p,held) for p in prompts],key=lambda a:average(support,a))
    ks, ns = average(support,known), average(support,new)
    # Compare on the same support queries. Ties retain the known policy.
    adapted = new if ns > ks else known
    return dict(fixed_known=known,fixed_new_support=new,fixed_adapted=adapted), dict(
        known_validation_R=average(validation,known), known_support_R=ks,new_support_R=ns,
        tie_break='first semantic action; adaptation ties retain known',
        sources='known pair: known validation only; new prompt and incorporation: reserved support only')


def fixed_metrics(result, action):
    pi, mi = result['prompt_order'].index(action[0]),result['model_order'].index(action[1])
    values = [q[pi][mi] for q in result['target_outcomes']]
    lam = result['config']['reward_weight']
    # target_outcomes follows selections/source-row order, not sorted test_qids.
    qids = [x['qid'] for x in result['results']['calibrated_8']['selections']]
    records = [dict(qid=qid,P=v[0],C=v[1],R=v[0]-lam*v[1],C_qnorm=v[2],R_qnorm=v[0]-lam*v[2])
               for qid,v in zip(qids,values)]
    return dict(action=dict(prompt=action[0],model=action[1]),
                **{k:sum(x[k] for x in records)/len(records) for k in ('P','C','R','C_qnorm','R_qnorm')},
                selections=records)


def paired(calibrated, baseline, seed):
    bm = {x['qid']:x['R'] for x in baseline['selections']}
    delta = [x['R']-bm[x['qid']] for x in calibrated['selections']]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(2000))
    return dict(gain=sum(delta)/len(delta),conditional_bootstrap_95_ci=[draws[49],draws[1949]],
                beneficial=sum(d>1e-7 for d in delta),harmful=sum(d< -1e-7 for d in delta),
                tied=sum(abs(d)<=1e-7 for d in delta))


def audit(result, rows):
    if result.get('protocol') != 'official_graphrouter_joint_unseen_v1':
        raise ValueError('not the contracted joint result')
    ids = [r['qid'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate qids')
    groups = [set(result[k]) for k in ('train_qids','support_qids','validation_qids','test_qids')]
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]) or set.union(*groups) != set(ids):
        raise ValueError('partition overlap or dataset universe mismatch')
    by_id = {r['qid']:r for r in rows}
    ref, cfg = result['cost_reference'], result['config']
    lo, span = ref['min'], ref['max']-ref['min']
    if not math.isfinite(lo) or not math.isfinite(span) or span<=0:
        raise ValueError('invalid monetary reference')
    validation = [by_id[q] for q in result['validation_qids']]
    support = [by_id[q] for q in result['support_qids']]
    policies, evidence = choose_policies(validation,support,result['prompt_order'],result['seen_models'],
                                         cfg['held_out_model'],lo,span,cfg['reward_weight'])
    # Freeze choices before reading saved target outcomes.
    frozen = json.dumps(policies,sort_keys=True)
    cal = result['results']['calibrated_8']
    qids = [x['qid'] for x in cal['selections']]
    if len(qids)!=len(set(qids)) or set(qids)!=groups[-1] or len(result['target_outcomes'])!=len(qids):
        raise ValueError('target order/coverage mismatch')
    if qids != [r['qid'] for r in rows if r['qid'] in groups[-1]]:
        raise ValueError('saved target order differs from original source rows')
    baselines = {k:fixed_metrics(result,v) for k,v in policies.items()}
    comparisons = {k:paired(cal,v,cfg['seed']) for k,v in baselines.items()}
    main = [comparisons[k] for k in ('fixed_known','fixed_adapted')]
    if any(x['gain'] <= 0 for x in main):
        verdict = 'no_advantage_over_both_fixed_controls'
    elif any(x['conditional_bootstrap_95_ci'][0] <= 0 for x in main):
        verdict = 'positive_point_gain_but_uncertain'
    else:
        verdict = 'conditional_advantage_on_this_development_fold'
    counts = cal['pair_counts']
    return dict(protocol='official_joint_fixed_policy_audit_v1',selection_evidence=evidence,
                frozen_policies_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
                baselines=baselines,comparisons=comparisons,verdict=verdict,
                calibrated_pair_counts=counts,
                instance_routing_advantage_demonstrated=False if len(counts)==1 else None,
                closure='Audit closes the current pilot. No automatic refit, variant or sweep.',
                limitation='Post-development comparator; one reused seed/fold. Bootstrap is conditional. Constant routing does not demonstrate instance adaptation.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results-path',type=Path,default=Path('outputs/graphrouter_joint_unseen/qwen2.5-14b/seed1_lam0.9/results.json'))
    ap.add_argument('--data-path',type=Path,required=True)
    args = ap.parse_args()
    out = args.results_path.with_name('fixed_policy_audit.json')
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    raw, source = args.data_path.read_bytes(), args.results_path.read_bytes()
    result = json.loads(source)
    if hashlib.sha256(raw).hexdigest() != result['dataset_sha256']:
        raise ValueError('dataset hash differs from the trained result')
    report = audit(result,[json.loads(l) for l in raw.decode().splitlines() if l.strip()])
    report.update(source_results_sha256=hashlib.sha256(source).hexdigest(),dataset_sha256=result['dataset_sha256'])
    with out.open('x') as f:
        f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verdict=report['verdict'],policies={k:v['action'] for k,v in report['baselines'].items()},
                         comparisons=report['comparisons']),indent=2))
    print(f'results={out}')


if __name__ == '__main__':
    main()

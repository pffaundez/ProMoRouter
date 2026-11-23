#!/usr/bin/env python3
"""One official-GraphRouter joint prompt/model held-out-model development run."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch
from torch.nn import functional as F
import torch_geometric

from router.data_validation import require_complete_action_space
from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_joint import (action_data, best_action_labels, cost_reference,
    graph_inputs, make_visible, evaluate_scores)
from router.splits import load_or_create_splits
from train_router_support_graph import (MODELS, PROMPTS, load_table, select_support,
    incorporation as joint_incorporation)
from train_graphrouter_unseen import macro_f1, UPSTREAM_SHA


def summarize(rows, scores, truth, pool, lam):
    selected = []
    for i, a in enumerate(scores.flatten(1).argmax(1).tolist()):
        pi, mi = divmod(a, len(pool))
        p, c, cq = map(float, truth[i,pi,mi])
        selected.append(dict(qid=rows[i]['qid'],task=rows[i]['task'],prompt=PROMPTS[pi],
                             model=pool[mi],P=p,C=c,R=p-lam*c,C_qnorm=cq,R_qnorm=p-lam*cq))
    def aggregate(items):
        counts = Counter(f"{x['prompt']}|{x['model']}" for x in items)
        n = len(items)
        return dict(queries=n, **{k:sum(x[k] for x in items)/n for k in ('P','C','R','C_qnorm','R_qnorm')},
                    model_counts=dict(Counter(x['model'] for x in items)),
                    prompt_counts=dict(Counter(x['prompt'] for x in items)), pair_counts=dict(counts),
                    top_pair_share=max(counts.values())/n,
                    pair_entropy=-sum((v/n)*math.log(v/n) for v in counts.values()))
    result = aggregate(selected)
    result.update(selections=selected, by_task={t:aggregate([x for x in selected if x['task']==t])
                  for t in sorted({x['task'] for x in selected})})
    return result


def compare_direct(path, payload):
    """Compare metadata only; no historical test outcomes influence fitting."""
    if not path.exists():
        return dict(status='unavailable', path=str(path), numerically_comparable=False)
    previous = json.loads(path.read_text())
    checks = dict(protocol=previous.get('protocol')=='official_graphrouter_direct_unseen_v1',
                  upstream_commit=previous.get('upstream_commit')==UPSTREAM_SHA)
    for k in ('dataset_sha256','train_qids','validation_qids','test_qids','seen_models'):
        checks[k] = previous.get(k)==payload[k]
    checks['support_qids'] = previous.get('small_support_qids')==payload['support_qids']
    for k in ('min','max','field','clipped'):
        checks['cost_'+k] = previous.get('cost_reference',{}).get(k)==payload['cost_reference'][k]
    for k in ('seed','reward_weight','held_out_model'):
        checks[k] = previous.get('config',{}).get(k)==payload['config'][k]
    return dict(status='verified' if all(checks.values()) else 'mismatch',
                path=str(path), artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                numerically_comparable=all(checks.values()), checks=checks,
                limitation='Same primary units only; action space, input width, history and support calls differ.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data','query-emb','task-emb','prompt-emb','model-emb'):
        ap.add_argument('--'+name+'-path', type=Path, required=True)
    ap.add_argument('--held-out-model', choices=MODELS, default='qwen2.5-14b')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--reward-weight', type=float, default=.9)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--split-manifest', type=Path)
    ap.add_argument('--direct-results-path', type=Path)
    ap.add_argument('--output-dir', type=Path, default=Path('outputs/graphrouter_joint_unseen'))
    args = ap.parse_args()
    if args.epochs < 1 or not math.isfinite(args.reward_weight) or args.reward_weight < 0:
        ap.error('positive epochs and finite nonnegative lambda required')
    out = args.output_dir/args.held_out_model/f'seed{args.seed}_lam{args.reward_weight}'
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    raw = args.data_path.read_bytes()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    if len({r['qid'] for r in rows}) != len(rows):
        raise ValueError('duplicate qids')
    require_complete_action_space(rows, PROMPTS, MODELS)
    manifest = args.split_manifest or Path(f'data/router/splits/qnorm_complete_seed{args.seed}.json')
    train, val, test = load_or_create_splits([r['qid'] for r in rows],manifest,args.seed)
    support_ids = select_support(train, 8, args.seed)
    take = lambda ids: [r for r in rows if r['qid'] in ids]
    fit, context, validation, targets, support = map(take,(train-support_ids,train,val,test,support_ids))
    seen = tuple(m for m in MODELS if m != args.held_out_model)
    lo, hi = cost_reference(fit,seen)
    # Structural validation above includes all IDs; outcomes are read only for
    # requested candidates. Held-out numeric outcomes remain unread until freeze.
    values = action_data(context,PROMPTS,seen,lo,hi-lo)
    vdata = action_data(validation,PROMPTS,seen,lo,hi-lo)
    visible = make_visible(context,PROMPTS,seen,values,args.reward_weight)
    tables = tuple(load_table(p) for p in (args.query_emb_path,args.task_emb_path,
                                         args.prompt_emb_path,args.model_emb_path))
    task, query, models, src, dst, attrs = graph_inputs(context,[],PROMPTS,seen,tables,visible,device)
    edges = torch.tensor([src,dst],dtype=torch.long,device=device)
    attributes = torch.tensor(attrs,dtype=torch.float32,device=device)
    eligible = torch.tensor([qid in (train-support_ids) for qid,_,_,_ in visible],dtype=torch.bool,device=device)
    _, labels = best_action_labels(values,args.reward_weight)
    labels = labels.to(device)
    vbest, _ = best_action_labels(vdata,args.reward_weight)
    net = EncoderDecoderNet(query.shape[1],models.shape[1],8,3).to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    payload = dict(protocol='official_graphrouter_joint_unseen_v1',upstream_commit=UPSTREAM_SHA,
                   config=config,dataset_sha256=hashlib.sha256(raw).hexdigest(),split_manifest=str(manifest),
                   train_qids=sorted(train-support_ids),support_qids=sorted(support_ids),
                   validation_qids=sorted(val),test_qids=sorted(test),seen_models=list(seen),
                   prompt_order=list(PROMPTS),model_order=list(MODELS),
                   cost_reference=dict(field='cost_proxy_money',min=lo,max=hi,
                                       fit='seen Direct fitting queries only',clipped=False))
    direct_path = args.direct_results_path or Path('outputs/graphrouter_unseen')/args.held_out_model/f'seed{args.seed}_lam{args.reward_weight}'/'results.json'
    payload['direct_reference'] = compare_direct(direct_path,payload)
    print('direct_reference='+json.dumps(payload['direct_reference']),flush=True)
    # A missing/mismatched historical reference does not invalidate the primary
    # within-checkpoint contrast; never report a matched cross-run comparison.
    out.mkdir(parents=True)
    best_f1, state, history = -1., None, []
    for epoch in range(1,args.epochs+1):
        net.train(); opt.zero_grad(set_to_none=True); loss_sum = 0.
        for _ in range(32):
            predict = eligible & (torch.rand(len(eligible),device=device)>=.5)
            if not bool(predict.any()):
                raise RuntimeError('empty prediction mask')
            scores = net(task,query,models,edges,edge_mask=predict,
                         edge_can_see=~predict,edge_weight=attributes)
            loss = F.binary_cross_entropy(scores.flatten(),labels[predict])
            (loss/32).backward(); loss_sum += float(loss.detach())/32
        opt.step()
        vs = evaluate_scores(net,context,validation,PROMPTS,seen,tables,visible,device)
        f1 = macro_f1(vbest,vs.flatten(1).argmax(1),len(PROMPTS)*len(seen))
        vm = summarize(validation,vs,vdata,seen,args.reward_weight)
        history.append(dict(epoch=epoch,loss=loss_sum,validation_macro_f1=f1,validation_R=vm['R']))
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch == 1 or epoch%10 == 0:
            print(f'epoch={epoch} loss={loss_sum:.6f} val_F1={f1:.6f} val_R={vm["R"]:.6f} best_epoch={best_epoch}',flush=True)
    net.load_state_dict(state); net.eval(); net.requires_grad_(False)
    # Outcomes below are offline evaluation/support only, after checkpoint freeze.
    truth = action_data(targets,PROMPTS,MODELS,lo,hi-lo)
    support_truth = action_data(support,PROMPTS,(args.held_out_model,),lo,hi-lo)
    new_edges = make_visible(support,PROMPTS,(args.held_out_model,),support_truth,args.reward_weight)
    result, saved_scores, graph_counts = {}, {}, {}
    for name in ('seen_only','description_only','calibrated_8'):
        pool = seen if name == 'seen_only' else MODELS
        observed = visible+new_edges if name == 'calibrated_8' else visible
        sc = evaluate_scores(net,context,targets,PROMPTS,pool,tables,observed,device)
        si = [pool.index(m) for m in seen]
        if name != 'seen_only' and not torch.allclose(sc[:,:,si],torch.tensor(saved_scores['seen_only']),atol=1e-6,rtol=1e-6):
            raise RuntimeError('new model changed known candidate scores')
        observed_truth = truth[:,:,[MODELS.index(m) for m in pool]]
        result[name] = summarize(targets,sc,observed_truth,pool,args.reward_weight)
        saved_scores[name] = sc.tolist()
        graph_counts[name] = dict(context_sources=len(context)*len(PROMPTS),
                                 target_sources=len(targets)*len(PROMPTS),models=len(pool),
                                 visible_edges=len(observed),prediction_edges=len(targets)*len(PROMPTS)*len(pool))
        if name != 'seen_only':
            inc = joint_incorporation(result[name],result['seen_only'],truth[...,:2],MODELS,args.held_out_model,args.reward_weight)
            delta = torch.tensor([a['R']-b['R'] for a,b in zip(result[name]['selections'],result['seen_only']['selections'])],dtype=torch.float64)
            g = torch.Generator().manual_seed(args.seed)
            draws = delta[torch.randint(len(delta),(2000,len(delta)),generator=g)].mean(1)
            inc['paired_bootstrap_95_ci'] = draws.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist()
            inc['gain_qnorm'] = result[name]['R_qnorm']-result['seen_only']['R_qnorm']
            result[name]['incorporation'] = inc
        print(name,json.dumps({k:result[name][k] for k in ('P','C','R','R_qnorm','model_counts','prompt_counts')}),flush=True)
    if any(not torch.equal(v,net.state_dict()[k].cpu()) for k,v in state.items()):
        raise RuntimeError('parameters or BatchNorm buffers changed during incorporation')
    payload.update(best_epoch=best_epoch,history=history,results=result,candidate_scores=saved_scores,
                   target_outcomes=truth.tolist(),graph_counts=graph_counts,
                   parameter_count=sum(v.numel() for v in net.parameters()),
                   input_dimensions=dict(query_prompt=query.shape[1],task=task.shape[1],model=models.shape[1],common=8),
                   training=dict(mask_draws=32,mask_probability=.5,epochs=args.epochs,
                                 positive_labels_per_query=1,optimizer='AdamW',lr=.0003,weight_decay=.0001),
                   calibration_calls=len(support)*len(PROMPTS),
                   calibration_money=sum(float(e['cost_proxy_money']) for r in support for e in r['action_edges'] if e['model']==args.held_out_model and e['prompt'] in PROMPTS),
                   support_outcomes=support_truth.tolist(),
                   calibration_gain=result['calibrated_8']['R']-result['description_only']['R'],
                   descriptive_incorporation_improved=result['calibrated_8']['R']>result['seen_only']['R'],
                   runtime=dict(torch=torch.__version__,pyg=torch_geometric.__version__,device=str(device),deterministic=True),
                   limitation='One reused development fold; no significance or net amortized monetary savings claim. Joint k8 costs 32 actions, Direct k8 costs eight.',
                   metric_contract='Primary frozen known Direct monetary normalization; stored qnorm is secondary only.')
    c = truth[...,1]
    payload['test_cost_extrapolation'] = dict(below_zero=int((c<0).sum()),above_one=int((c>1).sum()),minimum=float(c.min()),maximum=float(c.max()))
    torch.save(dict(state_dict=state,config=config,upstream_commit=UPSTREAM_SHA,
                    prompt_order=PROMPTS,model_order=MODELS,cost_reference=payload['cost_reference']),out/'router.pt')
    (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(f'results={out / "results.json"}',flush=True)


if __name__ == '__main__':
    main()

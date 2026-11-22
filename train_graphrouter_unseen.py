#!/usr/bin/env python3
"""Pinned official GraphRouter encoder; fixed-Direct held-out-LLM control."""
import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from router.graphrouter_official import EncoderDecoderNet
from router.splits import load_or_create_splits
from train_router_support_graph import MODELS, TASKS, load_table, vector, select_support

UPSTREAM_SHA = '0edfb8758d45acd93c8029354867ffb49bb8a97f'


def direct_data(rows, pool, cost_min, cost_span):
    """Raw money is normalized once with fixed training-only endpoints."""
    output = []
    for row in rows:
        direct = {e['model']: e for e in row['action_edges'] if e['prompt']=='direct'}
        output.append([[float(direct[m]['performance']),
                        (float(direct[m]['cost_proxy_money'])-cost_min)/cost_span,
                        float(direct[m]['cost_norm_query'])] for m in pool])
    return torch.tensor(output, dtype=torch.float32)


def nested_support(train_ids, small, large, seed):
    first = select_support(train_ids, small, seed)
    if large < small or large > len(train_ids):
        raise ValueError('large support must contain small support and fit in train')
    extra = random.Random(seed).sample(sorted(set(train_ids)-first), large-small)
    return first, first | set(extra)


def macro_f1(target, prediction, classes):
    values = []
    for c in range(classes):
        tp = int(((target==c)&(prediction==c)).sum())
        fp = int(((target!=c)&(prediction==c)).sum())
        fn = int(((target==c)&(prediction!=c)).sum())
        # sklearn's default includes labels present in y_true or y_pred.
        if 2*tp+fp+fn:
            values.append(2*tp/(2*tp+fp+fn))
    return sum(values)/len(values) if values else 0.


def graph_inputs(context, targets, pool, tables, visible, device):
    """Only context outcomes enter GNN edge attributes; target inputs are text."""
    qt, tt, mt = tables
    if {r['qid'] for r in context}&{r['qid'] for r in targets}:
        raise ValueError('context and target queries overlap')
    rows = context + targets
    query = torch.stack([vector(qt,r['qid'],()) for r in rows]).to(device)
    task = torch.stack([vector(tt,r['task'],TASKS) for r in rows]).to(device)
    model = torch.stack([vector(mt,m,MODELS) for m in pool]).to(device)
    qindex = {r['qid']: i for i,r in enumerate(rows)}
    sources, destinations, attrs = [], [], []
    for qid, name, attr in visible:
        if qid not in qindex or qindex[qid]>=len(context) or name not in pool:
            raise ValueError('edge outside permitted context/pool')
        sources.append(qindex[qid]); destinations.append(len(rows)+pool.index(name)); attrs.append(attr)
    return task, query, model, sources, destinations, attrs


def make_visible(context, pool, values, lam):
    return [(row['qid'], model, [float(values[i,j,0]-lam*values[i,j,1]),
                                 float(values[i,j,1]), float(values[i,j,0])])
            for i,row in enumerate(context) for j,model in enumerate(pool)]


def evaluate_scores(net, context, targets, pool, tables, visible, device):
    task, query, models, sources, dest, attrs = graph_inputs(context, targets, pool, tables, visible, device)
    count = len(sources)
    for i in range(len(context), len(context)+len(targets)):
        for m in range(len(pool)):
            sources.append(i); dest.append(len(query)+m)
    edges = torch.tensor([sources,dest],dtype=torch.long,device=device)
    attributes = torch.zeros(len(sources),3,device=device)
    if count: attributes[:count] = torch.tensor(attrs,device=device)
    can_see = torch.arange(len(sources),device=device)<count
    predict = ~can_see
    net.eval()
    with torch.inference_mode():
        result = net(task,query,models,edges,edge_mask=predict,
                     edge_can_see=can_see,edge_weight=attributes)
    return result.view(len(targets),len(pool)).cpu()


def summarize(rows, scores, data, pool, lam):
    selected = []
    for i,j in enumerate(scores.argmax(1).tolist()):
        p,c,cq = data[i,j].tolist()
        selected.append(dict(qid=rows[i]['qid'],task=rows[i]['task'],prompt='direct',
                             model=pool[j],P=p,C=c,R=p-lam*c,C_qnorm=cq,R_qnorm=p-lam*cq))
    summary = {k:sum(r[k] for r in selected)/len(selected) for k in ('P','C','R','C_qnorm','R_qnorm')}
    counts = Counter(r['model'] for r in selected)
    summary.update(queries=len(selected),model_counts=dict(counts),
                   prompt_counts={'direct':len(selected)},top_model_share=max(counts.values())/len(selected),
                   by_task={},selections=selected)
    for task in sorted({r['task'] for r in selected}):
        part = [r for r in selected if r['task']==task]
        summary['by_task'][task] = dict(queries=len(part),model_counts=dict(Counter(r['model'] for r in part)),
            **{k:sum(r[k] for r in part)/len(part) for k in ('P','C','R','R_qnorm')})
    return summary


def incorporation(full, baseline, truth, pool, held, lam):
    reward = truth[:,:,0]-lam*truth[:,:,1]
    h = pool.index(held); seen = [i for i in range(len(pool)) if i!=h]
    best_seen = reward[:,seen].max(1).values
    opportunities = reward[:,h]>best_seen+1e-7
    rq = truth[:,:,0]-lam*truth[:,:,2]
    qseen = rq[:,seen].max(1).values
    delta = [a['R']-b['R'] for a,b in zip(full['selections'],baseline['selections'])]
    chosen = [r['model']==held for r in full['selections']]
    g = torch.Generator().manual_seed(1)
    d = torch.tensor(delta,dtype=torch.float64)
    means = d[torch.randint(len(d),(2000,len(d)),generator=g)].mean(1)
    return dict(gain=full['R']-baseline['R'],gain_qnorm=full['R_qnorm']-baseline['R_qnorm'],
                paired_bootstrap_95_ci=means.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
                oracle_gain=float((reward.max(1).values-best_seen).mean()),oracle_wins=int(opportunities.sum()),
                oracle_gain_qnorm=float((rq.max(1).values-qseen).mean()),
                oracle_wins_qnorm=int((rq[:,h]>qseen+1e-7).sum()),
                held_out_selections=sum(chosen),model_hits=sum(c and bool(o) for c,o in zip(chosen,opportunities)),
                beneficial=sum(c and d>1e-7 for c,d in zip(chosen,delta)),
                harmful=sum(c and d < -1e-7 for c,d in zip(chosen,delta)),
                tied=sum(c and abs(d)<=1e-7 for c,d in zip(chosen,delta)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','query-emb','task-emb','model-emb'):
        ap.add_argument('--'+name+'-path',type=Path,required=True)
    ap.add_argument('--held-out-model',choices=MODELS,default='qwen2.5-14b')
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--reward-weight',type=float,default=.9)
    ap.add_argument('--small-support',type=int,default=8)
    ap.add_argument('--large-support',type=int,default=80)
    ap.add_argument('--epochs',type=int,default=1000)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--split-manifest',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/graphrouter_unseen'))
    args=ap.parse_args()
    if args.epochs<1 or not math.isfinite(args.reward_weight) or args.reward_weight<0:
        ap.error('epochs must be positive and lambda finite/nonnegative')
    out=args.output_dir/args.held_out_model/f'seed{args.seed}_lam{args.reward_weight}'
    if out.exists(): raise FileExistsError(f'refusing to overwrite {out}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(args.device)
    raw=args.data_path.read_bytes()
    rows=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    if len({r['qid'] for r in rows})!=len(rows):raise ValueError('duplicate qids')
    for row in rows:
        direct=[e for e in row['action_edges'] if e['prompt']=='direct']
        if len(direct)!=len(MODELS) or {e['model'] for e in direct}!=set(MODELS):
            raise ValueError('incomplete or duplicated Direct pool')
        for e in direct:
            p,c,cq=e['performance'],e.get('cost_proxy_money'),e['cost_norm_query']
            if c is None or not all(math.isfinite(float(v)) for v in (p,c,cq)) or not (0<=p<=1 and c>=0 and 0<=cq<=1):
                raise ValueError('valid cost_proxy_money, performance and cost_norm_query required; no implicit cost fallback')
    manifest=args.split_manifest or Path(f'data/router/splits/qnorm_complete_seed{args.seed}.json')
    train,val,test=load_or_create_splits([r['qid'] for r in rows],manifest,args.seed)
    small,large=nested_support(train,args.small_support,args.large_support,args.seed)
    take=lambda ids:[r for r in rows if r['qid'] in ids]
    fit,support,validation,targets=map(take,(train-small,small,val,test))
    context=take(train)
    seen=tuple(m for m in MODELS if m!=args.held_out_model)
    # Only actual fitting qids and seen Direct candidates fit the cost reference.
    costs=[float(e['cost_proxy_money']) for r in fit for e in r['action_edges']
           if e['prompt']=='direct' and e['model'] in seen]
    lo,hi=min(costs),max(costs)
    if hi<=lo:raise ValueError('degenerate training monetary cost reference')
    span=hi-lo
    values=direct_data(context,seen,lo,span)
    lam=args.reward_weight
    visible=make_visible(context,seen,values,lam)
    tables=tuple(load_table(p) for p in (args.query_emb_path,args.task_emb_path,args.model_emb_path))
    task,query,models,src,dst,attrs=graph_inputs(context,[],seen,tables,visible,device)
    edges=torch.tensor([src,dst],dtype=torch.long,device=device)
    attributes=torch.tensor(attrs,dtype=torch.float32,device=device)
    eligible=torch.tensor([qid in (train-small) for qid,_,_ in visible],dtype=torch.bool,device=device)
    best=(values[:,:,0]-lam*values[:,:,1]).argmax(1)
    labels=F.one_hot(best,num_classes=len(seen)).flatten().float().to(device)
    # Match config.yaml embedding_dim=8: official internal width is 2*8=16.
    net=EncoderDecoderNet(query.shape[1],models.shape[1],8,3).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    vdata=direct_data(validation,seen,lo,span)
    vbest=(vdata[:,:,0]-lam*vdata[:,:,1]).argmax(1)
    best_f1=-1.;state=None;history=[]
    for epoch in range(1,args.epochs+1):
        net.train();opt.zero_grad(set_to_none=True);loss_sum=0.
        # Upstream 'batch_size=32' means 32 random masks per optimizer step,
        # not 32-query minibatches. Backprop each /32 loss to limit memory.
        for _ in range(32):
            predict=eligible & (torch.rand(len(eligible),device=device)>=.5)
            if not bool(predict.any()):raise RuntimeError('empty sampled prediction mask')
            see=~predict
            pred=net(task,query,models,edges,edge_mask=predict,edge_can_see=see,edge_weight=attributes)
            loss=F.binary_cross_entropy(pred.flatten(),labels[predict])
            (loss/32).backward();loss_sum+=float(loss.detach())/32
        opt.step()
        vs=evaluate_scores(net,context,validation,seen,tables,visible,device)
        f1=macro_f1(vbest,vs.argmax(1),len(seen))
        history.append(dict(epoch=epoch,loss=loss_sum,validation_macro_f1=f1))
        if f1>best_f1:
            best_f1=f1;best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch==1 or epoch%10==0:print(f'epoch={epoch} loss={loss_sum:.6f} val_F1={f1:.5f} best_epoch={best_epoch}',flush=True)
    net.load_state_dict(state);net.eval();net.requires_grad_(False)
    truth=direct_data(targets,MODELS,lo,span)
    result={};scores={}
    for name,ids in (('seen_only',None),('description_only',set()),('calibrated_8',small),('calibrated_80',large)):
        pool=seen if ids is None else MODELS
        observed=list(visible)
        if ids:
            extra=take(ids)
            observed+=make_visible(extra,(args.held_out_model,),direct_data(extra,(args.held_out_model,),lo,span),lam)
        sc=evaluate_scores(net,context,targets,pool,tables,observed,device)
        scores[name]=sc.tolist()
        data=truth[:,[MODELS.index(m) for m in pool]]
        result[name]=summarize(targets,sc,data,pool,lam)
        if ids is not None:
            result[name]['incorporation']=incorporation(result[name],result['seen_only'],truth,MODELS,args.held_out_model,lam)
            h=MODELS.index(args.held_out_model)
            seen_indices=[i for i in range(len(MODELS)) if i!=h]
            result[name]['held_out_score_summary']=dict(
                mean_score=float(sc[:,h].mean()),
                median_rank=float((1+(sc>sc[:,h,None]).sum(1)).float().quantile(.5)),
                mean_margin_to_seen=float((sc[:,h]-sc[:,seen_indices].max(1).values).mean()))
        print(name,json.dumps({k:result[name][k] for k in ('P','C','R','R_qnorm','model_counts')}),result[name].get('incorporation',''),flush=True)
    for key,value in state.items():
        if not torch.equal(value,net.state_dict()[key].cpu()):raise RuntimeError('parameters or BN buffers changed in inference')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    payload=dict(protocol='official_graphrouter_direct_unseen_v1',upstream_commit=UPSTREAM_SHA,
                 config=config,dataset_sha256=hashlib.sha256(raw).hexdigest(),split_manifest=str(manifest),
                 train_qids=sorted(train-small),small_support_qids=sorted(small),large_support_qids=sorted(large),
                 validation_qids=sorted(val),test_qids=sorted(test),seen_models=seen,
                 cost_reference=dict(field='cost_proxy_money',min=lo,max=hi,fit='seen Direct fitting queries only',clipped=False),
                 metric_contract='Primary R=P-lambda*C with frozen training money normalization. R_qnorm is historical stored full-pool cost reference; do not conflate.',
                 best_epoch=best_epoch,history=history,results=result,candidate_scores=scores,
                 target_outcomes_direct=truth.tolist(),target_outcome_order=list(MODELS),
                 calibration_calls=dict(small=len(small),large=len(large)),
                 calibration_gain={k:result[k]['R']-result['description_only']['R'] for k in ('calibrated_8','calibrated_80')},
                 calibration_money={k:sum(float(e['cost_proxy_money']) for r in take(ids) for e in r['action_edges'] if e['prompt']=='direct' and e['model']==args.held_out_model)
                                    for k,ids in (('small',small),('large',large))},
                 runtime=dict(torch=torch.__version__,device=str(device),deterministic=True),
                 limitation='Adapted official-code control, one development fold; bootstrap is conditional on this checkpoint, not across training seeds.')
    out.mkdir(parents=True)
    torch.save(dict(state_dict=state,config=config,upstream_commit=UPSTREAM_SHA),out/'router.pt')
    (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(f'results={out / "results.json"}')


if __name__=='__main__':main()

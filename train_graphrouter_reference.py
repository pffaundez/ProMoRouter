#!/usr/bin/env python3
"""Few-shot reference reconstructed from pinned public GraphRouter + paper Sec 3.2."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import torch
from torch.nn import functional as F
import torch_geometric

from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_reference import MODELS, UPSTREAM, predict
from train_graphrouter_unseen import macro_f1


def training_graph(data,device):
    """Keep the upstream full-edge ordering/RNG layout, with zero hidden attrs."""
    q=data['query'].to(device);t=data['task'].to(device);m=data['model'][:6].to(device)
    n=len(q);pool=6
    src=torch.arange(n,device=device).repeat_interleave(pool)
    dst=(torch.arange(pool,device=device)+n).repeat(n)
    edge_index=torch.stack((src,dst))
    train=data['splits']['train']
    mask=torch.zeros(n,pool,dtype=torch.bool,device=device);mask[train]=True
    attrs=torch.zeros(n,pool,3,device=device)
    p=data['performance'][train,:6].to(device);c=data['cost'][train,:6].to(device)
    reward=.5*p-.5*c
    attrs[train]=torch.stack((reward,c,p),-1)
    labels=torch.zeros(n,pool,device=device)
    labels[train]=F.one_hot(reward.argmax(1),num_classes=6).float()
    return t,q,m,edge_index,attrs.flatten(0,1),mask.flatten(),labels.flatten()


def train_step(net,opt,inputs,draws=32):
    t,q,m,edges,attrs,eligible,labels=inputs
    net.train();opt.zero_grad(set_to_none=True)
    losses=[]
    for _ in range(draws):
        # Upstream samples masks on CPU even when the network is on CUDA.
        # Keep that RNG stream instead of silently switching to a CUDA generator.
        random_mask=(torch.rand(len(eligible),device='cpu')<.5).to(eligible.device)
        predict_mask=eligible & ~random_mask
        see=~predict_mask & eligible
        if not bool(predict_mask.any()):raise RuntimeError('empty training edge sample')
        scores=net(t,q,m,edges,edge_mask=predict_mask,edge_can_see=see,edge_weight=attrs)
        losses.append(F.binary_cross_entropy(scores.flatten(),labels[predict_mask]))
    loss=torch.stack(losses).mean()
    loss.backward();opt.step()
    return float(loss.detach())


def summarize(data,targets,pool,scores):
    chosen=scores.argmax(1).tolist();records=[]
    for qi,j in zip(targets,chosen):
        mi=pool[j];p=float(data['performance'][qi,mi]);c=float(data['cost'][qi,mi])
        records.append(dict(query_index=qi,task=data['task_names'][qi],model=MODELS[mi],
                            P=p,C=c,R=.5*p-.5*c))
    def metrics(rs):
        counts=Counter(x['model'] for x in rs)
        return dict(queries=len(rs),**{k:sum(x[k] for x in rs)/len(rs) for k in ('P','C','R')},
                    model_counts=dict(counts),top_model_share=max(counts.values())/len(rs))
    summary=metrics(records)
    summary.update(selections=records,by_task={t:metrics([x for x in records if x['task']==t])
                   for t in sorted({x['task'] for x in records})})
    return summary


def gains(full,base,seed):
    delta=torch.tensor([x['R']-y['R'] for x,y in zip(full['selections'],base['selections'])],dtype=torch.float64)
    g=torch.Generator().manual_seed(seed)
    samples=delta[torch.randint(len(delta),(2000,len(delta)),generator=g)].mean(1)
    return dict(gain=float(delta.mean()),conditional_bootstrap_95_ci=samples.quantile(torch.tensor([.025,.975],dtype=torch.float64)).tolist(),
                beneficial=int((delta>1e-7).sum()),harmful=int((delta< -1e-7).sum()),tied=int((delta.abs()<=1e-7).sum()))


def run(data,epochs,device,out):
    meta=data['metadata'];synthetic=meta.get('mode')=='synthetic_engineering'
    if not synthetic and (not meta.get('authors_release_verified') or meta.get('model_names')!=list(MODELS)):
        raise ValueError('reference dataset validation failed; no ProMoRouter fallback')
    seed=1881
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    out=Path(out)
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    device=torch.device(device)
    inputs=training_graph(data,device)
    t,q,m,_,_,_,_=inputs
    net=EncoderDecoderNet(q.shape[1],m.shape[1],8,3).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    split=data['splits'];train,val,test,support=(split[k] for k in ('train','validation','test','support'))
    known=tuple(range(6));all_models=tuple(range(10))
    vtruth=.5*data['performance'][val,:6]-.5*data['cost'][val,:6]
    vbest=vtruth.argmax(1)
    state=None;best=-1.;history=[]
    out.mkdir(parents=True)
    for epoch in range(1,epochs+1):
        loss=train_step(net,opt,inputs)
        scores=predict(net,data,known,train,[known]*len(train),val,device)
        f1=macro_f1(vbest,scores.argmax(1),6)
        vm=summarize(data,val,known,scores)
        history.append(dict(epoch=epoch,loss=loss,validation_macro_f1=f1,validation_R=vm['R']))
        if f1>best:
            best,best_epoch=f1,epoch
            state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch==1 or epoch%10==0:
            print(f'epoch={epoch} loss={loss:.6f} val_F1={f1:.6f} val_R={vm["R"]:.6f} best_epoch={best_epoch}',flush=True)
    best_state=state
    # Upstream test() does not reload the saved best-validation checkpoint.
    # Retain that exact primary behavior, without its epoch-wise test calls.
    state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    net.eval();net.requires_grad_(False)
    # Public test() uses known train+validation history. No unseen validation
    # outcome is ever revealed. Checkpoint selection has completed already.
    history_queries=train+val
    history_models=[known]*len(history_queries)
    results,saved,counts={},{},{}
    for name in ('known_only','description_only','fewshot_80'):
        pool=known if name=='known_only' else all_models
        vq=list(history_queries);vmodels=list(history_models)
        if name=='fewshot_80':
            vq+=support;vmodels+=[tuple(range(6,10))]*len(support)
        sc=predict(net,data,pool,vq,vmodels,test,device)
        if name!='known_only' and not torch.allclose(sc[:,:6],torch.tensor(saved['known_only']),atol=1e-6,rtol=1e-6):
            raise RuntimeError('new nodes changed known-model scores')
        saved[name]=sc.tolist();results[name]=summarize(data,test,pool,sc)
        counts[name]=dict(query_feature_nodes=len(data['query']),model_nodes=len(pool),
                          visible_known_edges=len(history_queries)*6,
                          visible_new_edges=0 if name!='fewshot_80' else len(support)*4)
        if name!='known_only':results[name]['vs_known']=gains(results[name],results['known_only'],seed)
        print(name,json.dumps({k:results[name][k] for k in ('P','C','R','model_counts')}),flush=True)
    if any(not torch.equal(v,net.state_dict()[k].cpu()) for k,v in state.items()):
        raise RuntimeError('parameters/BatchNorm buffers changed in few-shot adaptation')
    # A fixed model is selected solely on known validation, not unseen labels.
    fixed=int(vtruth.mean(0).argmax())
    dummy=torch.zeros(len(test),6);dummy[:,fixed]=1.
    results['fixed_known_validation']=summarize(data,test,known,dummy)
    result=dict(protocol='graphrouter_public_data_fewshot_reference_v1',upstream_commit=UPSTREAM,
                provenance=meta,seed=seed,scenario='Balance',reward='0.5*effect-0.5*authors_stored_task_normalized_cost',
                model_names=list(MODELS),known_models=list(MODELS[:6]),unseen_models=list(MODELS[6:]),
                splits=split,best_epoch=best_epoch,evaluation_epoch=epochs,history=history,results=results,candidate_scores=saved,
                graph_counts=counts,calibration_queries=len(support),calibration_calls=len(support)*4,
                adaptation_frozen_state_verified=True,
                parameter_count=sum(v.numel() for v in net.parameters()),
                runtime=dict(torch=torch.__version__,pyg=torch_geometric.__version__,device=str(device)),
                training=dict(epochs=epochs,common_dim=8,internal_width=16,optimizer='AdamW',lr=.0003,
                              weight_decay=.0001,mask_draws=32,mask_rate=.5,
                              query_feature_policy='all CSV query embeddings participate in training BN as in public code',
                              test_history_policy='known train+validation, matching public test(); new support only after freeze',
                              checkpoint_policy='public test() evaluates final state; best-validation checkpoint saved separately, no epoch-wise test'),
                publication_reference=dict(fewshot_R=.207,trained_R=.219,quantitative_reproduction_verified=False,
                                           difference_to_published_fewshot=results['fewshot_80']['R']-.207,
                                           note='Report measured value and public-code/paper discrepancies; do not treat exact paper match as established.'),
                next_step='Inspect reference result and fidelity gaps before any prompt extension. No automated ProMoRouter launch.')
    result['fewshot_vs_description']=gains(results['fewshot_80'],results['description_only'],seed)
    result['fewshot_vs_fixed_known']=gains(results['fewshot_80'],results['fixed_known_validation'],seed)
    truth=.5*data['performance'][test]-.5*data['cost'][test]
    result['oracle_R']=float(truth.max(1).values.mean())
    result['oracle_new_model_opportunities']=int((truth[:,6:].max(1).values>truth[:,:6].max(1).values+1e-7).sum())
    torch.save(dict(state_dict=state,metadata=meta,splits=split,seed=seed,evaluation_epoch=epochs),out/'router.pt')
    torch.save(dict(state_dict=best_state,metadata=meta,splits=split,seed=seed,best_epoch=best_epoch),out/'best_validation_router.pt')
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'results={out / "results.json"}',flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared-path',type=Path,default=Path('data/graphrouter_reference/prepared.pt'))
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/graphrouter_reference/seed1881_balance'))
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--epochs',type=int,default=1000)
    args=ap.parse_args()
    if args.epochs<1:ap.error('epochs must be positive')
    data=torch.load(args.prepared_path,map_location='cpu',weights_only=True)
    run(data,args.epochs,args.device,args.output_dir)


if __name__=='__main__':main()

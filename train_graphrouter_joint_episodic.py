#!/usr/bin/env python3
"""Paired 8-model training: full-history control vs pseudo-new incorporation.

The real ninth model is excluded from both fits and added only after freeze.
"""
import argparse
import copy
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
from router.episodic_incorporation import EpisodeSampler
from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_joint import (action_data,best_action_labels,cost_reference,
                                      graph_inputs,make_visible,evaluate_scores)
from router.splits import load_or_create_splits
from train_graphrouter_joint_unseen import summarize
from train_graphrouter_unseen import macro_f1,UPSTREAM_SHA
from train_router_support_graph import MODELS,PROMPTS,load_table,select_support,incorporation
from audit_graphrouter_joint_fixed import choose_policies,fixed_metrics,paired


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        v=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def fit_pair(task,query,models,edges,attrs,labels,context,fitting_indices,validation,
             validation_truth,seen,tables,visible,args):
    device=torch.device(args.device)
    base=EncoderDecoderNet(query.shape[1],models.shape[1],8,3).to(device)
    nets={'control':base,'episodic':copy.deepcopy(base)}
    initial={name:state_hash(net.state_dict()) for name,net in nets.items()}
    if len(set(initial.values()))!=1:raise RuntimeError('paired models do not share initialization')
    opts={name:torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001) for name,net in nets.items()}
    best={name:-1. for name in nets};states={};histories={name:[] for name in nets};epochs={}
    sampler=EpisodeSampler(len(context),fitting_indices,len(PROMPTS),len(seen),args.seed,8)
    valbest,_=best_action_labels(validation_truth,args.reward_weight)
    examples=[];role_counts=Counter()
    for epoch in range(1,args.epochs+1):
        losses={name:0. for name in nets};roles=Counter();target_counts=[]
        for name,net in nets.items():net.train();opts[name].zero_grad(set_to_none=True)
        for draw in range(32):
            episode=sampler.sample();pseudo=episode['pseudo_index']
            roles[seen[pseudo]]+=1;role_counts[seen[pseudo]]+=1
            target_counts.append(int(episode['target_queries'].sum()))
            if epoch==1 and draw<8:
                examples.append(dict(pseudo_model=seen[pseudo],
                    support_qids=[context[i]['qid'] for i in episode['support_indices']],
                    target_qids=[r['qid'] for i,r in enumerate(context) if bool(episode['target_queries'][i])],
                    observed_actions_per_model=dict(zip(seen,episode['visible_per_model']))))
            predict=episode['predict'].to(device)
            for name,net in nets.items():
                see=episode[name+'_visible'].to(device)
                scores=net(task,query,models,edges,edge_mask=predict,edge_can_see=see,edge_weight=attrs)
                loss=F.binary_cross_entropy(scores.flatten(),labels[predict])
                (loss/32).backward();losses[name]+=float(loss.detach())/32
        if any(roles[m]!=4 for m in seen):raise RuntimeError('pseudo-new roles are not balanced per epoch')
        for name,net in nets.items():
            opts[name].step()
            vs=evaluate_scores(net,context,validation,PROMPTS,seen,tables,visible,device)
            f1=macro_f1(valbest,vs.flatten(1).argmax(1),len(PROMPTS)*len(seen))
            vr=summarize(validation,vs,validation_truth,seen,args.reward_weight)['R']
            histories[name].append(dict(epoch=epoch,loss=losses[name],validation_macro_f1=f1,
                                        validation_R=vr,mean_target_queries=sum(target_counts)/32))
            if f1>best[name]:
                best[name]=f1;epochs[name]=epoch
                states[name]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if epoch==1 or epoch%10==0:
            print('epoch='+str(epoch)+' '+json.dumps({name:histories[name][-1] for name in nets}),flush=True)
    for name,net in nets.items():
        net.load_state_dict(states[name]);net.eval();net.requires_grad_(False)
    return nets,dict(initial_state_sha256=initial,best_epochs=epochs,histories=histories,
                     episode_examples=examples,episode_plan_sha256=sampler.plan_sha256,
                     pseudo_model_role_counts=dict(role_counts),episode_draws=sampler.draws,
                     selected_state_sha256={name:state_hash(state) for name,state in states.items()})


def evaluate_pair(nets,context,support,targets,seen,tables,visible,lo,hi,args):
    # First and only held-out numerical reads occur after BOTH fits are frozen.
    if any(p.requires_grad for net in nets.values() for p in net.parameters()):
        raise RuntimeError('evaluation started before freeze')
    truth=action_data(targets,PROMPTS,MODELS,lo,hi-lo)
    new_truth=action_data(support,PROMPTS,(args.held_out_model,),lo,hi-lo)
    extra=make_visible(support,PROMPTS,(args.held_out_model,),new_truth,args.reward_weight)
    arms={}
    for name,net in nets.items():
        before=state_hash(net.state_dict());conditions={};scores_by_condition={};counts={}
        for condition in ('seen_only','description_only','calibrated_8'):
            pool=seen if condition=='seen_only' else MODELS
            observed=visible+extra if condition=='calibrated_8' else visible
            scores=evaluate_scores(net,context,targets,PROMPTS,pool,tables,observed,args.device)
            if condition!='seen_only':
                si=[pool.index(m) for m in seen]
                if not torch.allclose(scores[:,:,si],torch.tensor(scores_by_condition['seen_only']),atol=1e-6,rtol=1e-6):
                    raise RuntimeError('adding unseen candidate changed known scores')
            observed_truth=truth[:,:,[MODELS.index(m) for m in pool]]
            conditions[condition]=summarize(targets,scores,observed_truth,pool,args.reward_weight)
            scores_by_condition[condition]=scores.tolist()
            counts[condition]=dict(context_sources=len(context)*4,target_sources=len(targets)*4,
                                   models=len(pool),visible_edges=len(observed))
            if condition!='seen_only':
                conditions[condition]['incorporation']=incorporation(conditions[condition],conditions['seen_only'],truth[...,:2],MODELS,args.held_out_model,args.reward_weight)
                conditions[condition]['vs_seen']=paired(conditions[condition],conditions['seen_only'],args.seed)
        if state_hash(net.state_dict())!=before:raise RuntimeError('weights or BN buffers changed in adaptation')
        sc=torch.tensor(scores_by_condition['calibrated_8']);h=MODELS.index(args.held_out_model)
        si=[MODELS.index(m) for m in seen]
        margins=sc[:,:,h].max(1).values-sc[:,:,si].flatten(1).max(1).values
        arms[name]=dict(results=conditions,candidate_scores=scores_by_condition,graph_counts=counts,
                       frozen_state_sha256=before,calibration_gain=conditions['calibrated_8']['R']-conditions['description_only']['R'],
                       score_summary=dict(mean_new_margin=float(margins.mean()),min_new_margin=float(margins.min()),max_new_margin=float(margins.max())),
                       target_outcomes=truth.tolist(),support_outcomes=new_truth.tolist())
    return arms


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','query-emb','task-emb','prompt-emb','model-emb'):
        ap.add_argument('--'+name+'-path',type=Path,required=True)
    ap.add_argument('--held-out-model',choices=MODELS,default='qwen2.5-14b')
    ap.add_argument('--seed',type=int,default=1)
    ap.add_argument('--reward-weight',type=float,default=.9)
    ap.add_argument('--epochs',type=int,default=1000)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--split-manifest',type=Path)
    ap.add_argument('--output-dir',type=Path,default=Path('outputs/graphrouter_joint_episodic'))
    args=ap.parse_args()
    if args.epochs<1 or not math.isfinite(args.reward_weight) or args.reward_weight<0:ap.error('invalid epochs/lambda')
    if args.device.startswith('cuda') and not torch.cuda.is_available():ap.error('CUDA unavailable; no CPU fallback for the GPU runner')
    out=args.output_dir/args.held_out_model/f'seed{args.seed}_lam{args.reward_weight}'
    if out.exists():raise FileExistsError(f'refusing to overwrite {out}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.manual_seed(args.seed);random.seed(args.seed);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    raw=args.data_path.read_bytes();rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len({r['qid'] for r in rows})!=len(rows):raise ValueError('duplicate qids')
    require_complete_action_space(rows,PROMPTS,MODELS)
    manifest=args.split_manifest or Path(f'data/router/splits/qnorm_complete_seed{args.seed}.json')
    train,val,test=load_or_create_splits([r['qid'] for r in rows],manifest,args.seed)
    support_ids=select_support(train,8,args.seed);take=lambda ids:[r for r in rows if r['qid'] in ids]
    context,fit,validation,targets,support=map(take,(train,train-support_ids,val,test,support_ids))
    seen=tuple(m for m in MODELS if m!=args.held_out_model)
    lo,hi=cost_reference(fit,seen)
    values=action_data(context,PROMPTS,seen,lo,hi-lo)
    vtruth=action_data(validation,PROMPTS,seen,lo,hi-lo)
    visible=make_visible(context,PROMPTS,seen,values,args.reward_weight)
    tables=tuple(load_table(p) for p in (args.query_emb_path,args.task_emb_path,args.prompt_emb_path,args.model_emb_path))
    task,query,models,src,dst,attrs=graph_inputs(context,[],PROMPTS,seen,tables,visible,args.device)
    edges=torch.tensor([src,dst],dtype=torch.long,device=args.device)
    attributes=torch.tensor(attrs,dtype=torch.float32,device=args.device)
    _,labels=best_action_labels(values,args.reward_weight);labels=labels.to(args.device)
    fitting_indices=[i for i,r in enumerate(context) if r['qid'] not in support_ids]
    out.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    print('device='+str(args.device)+' models_seen='+str(seen)+' unseen='+args.held_out_model,flush=True)
    nets,training=fit_pair(task,query,models,edges,attributes,labels,context,fitting_indices,validation,vtruth,seen,tables,visible,args)
    arms=evaluate_pair(nets,context,support,targets,seen,tables,visible,lo,hi,args)
    # Same fixed controls, chosen without held-out validation or test outcomes.
    policies,evidence=choose_policies(validation,support,PROMPTS,seen,args.held_out_model,lo,hi-lo,args.reward_weight)
    view=dict(prompt_order=list(PROMPTS),model_order=list(MODELS),test_qids=sorted(test),config=config,
              target_outcomes=arms['episodic']['target_outcomes'],results=arms['episodic']['results'])
    controls={name:fixed_metrics(view,action) for name,action in policies.items()}
    for arm in arms.values():
        arm['vs_fixed']={name:paired(arm['results']['calibrated_8'],summary,args.seed) for name,summary in controls.items()}
    ep,co=arms['episodic']['results'],arms['control']['results']
    comparison=dict(calibrated=paired(ep['calibrated_8'],co['calibrated_8'],args.seed),
                    seen_only=paired(ep['seen_only'],co['seen_only'],args.seed))
    virtual=lambda arm:dict(selections=[dict(qid=a['qid'],R=a['R']-b['R']) for a,b in
        zip(arm['calibrated_8']['selections'],arm['seen_only']['selections'])])
    comparison['difference_in_incorporation_gain']=paired(virtual(ep),virtual(co),args.seed)
    flags=dict(improves_calibrated_over_control=comparison['calibrated']['gain']>0,
               positive_own_incorporation=ep['calibrated_8']['vs_seen']['gain']>0,
               improves_incorporation_over_control=comparison['difference_in_incorporation_gain']['gain']>0,
               beats_fixed_known=arms['episodic']['vs_fixed']['fixed_known']['gain']>0,
               beats_fixed_adapted=arms['episodic']['vs_fixed']['fixed_adapted']['gain']>0,
               constant_pair_policy=len(ep['calibrated_8']['pair_counts'])==1)
    relevant=[v for k,v in flags.items() if k!='constant_pair_policy']
    verdict='positive_descriptive_signal' if all(relevant) else 'not_supported_by_this_pilot'
    runtime=dict(torch=torch.__version__,pyg=torch_geometric.__version__,device=args.device,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),deterministic=True)
    if args.device.startswith('cuda'):runtime['gpu_name']=torch.cuda.get_device_name(torch.device(args.device))
    payload=dict(protocol='joint_pseudo_new_paired_v1',config=config,upstream_commit=UPSTREAM_SHA,
                 dataset_sha256=hashlib.sha256(raw).hexdigest(),split_manifest=str(manifest),
                 train_qids=sorted(train-support_ids),support_qids=sorted(support_ids),validation_qids=sorted(val),test_qids=sorted(test),
                 seen_models=list(seen),held_out_model=args.held_out_model,prompt_order=list(PROMPTS),model_order=list(MODELS),
                 cost_reference=dict(field='cost_proxy_money',min=lo,max=hi,fit='seen Direct fitting queries only',clipped=False),
                 training=training,arms=arms,fixed_controls=controls,fixed_selection_evidence=evidence,
                 comparison=comparison,flags=flags,verdict=verdict,runtime=runtime,
                 calibration_actions_per_arm=32,unique_held_out_support_actions_reused=32,
                 training_contract=dict(mask_unit='query, grouped over all prompts/models in both arms',
                     mask_probability=.5,mask_draws_per_epoch=32,support_queries_per_episode=8,
                     pseudo_roles_per_known_model_per_epoch=4,optimizer='AdamW',lr=.0003,weight_decay=.0001,
                     common_dim=8,epochs=args.epochs,loss='BCE; one positive joint action per original training query',
                     checkpoint='best known-only validation macro-F1 for each arm; no held-out tuning'),
                 historical_comparison='Old edge-masked checkpoint is historical only; primary is this matched query-masked control.',
                 limitation='One reused Qwen development fold, one seed, conditional bootstrap. Constant-pair routing does not establish instance adaptation.',
                 closure='Interpret this paired result and fixed controls before any further method change; no automatic sweep or retraining with the ninth model.')
    for name,net in nets.items():
        torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},config=config,
                        arm=name,seen_models=seen,cost_reference=payload['cost_reference'],best_epoch=training['best_epochs'][name]),out/f'{name}.pt')
    (out/'results.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verdict=verdict,flags=flags,comparison=comparison),indent=2),flush=True)
    print(f'results={out / "results.json"}',flush=True)


if __name__=='__main__':main()

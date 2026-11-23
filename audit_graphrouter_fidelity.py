#!/usr/bin/env python3
"""Bounded fidelity audit. No benchmark retraining or parameter search."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import urllib.request
import warnings

import torch
import torch_geometric
import sklearn
from torch.nn import functional as F

from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_reference import UPSTREAM, MODELS, RELEASE_CSV_SHA256, partitions
from train_graphrouter_reference import training_graph, train_step

SOURCE_BLOBS = {
    'graph_nn.py': ('model/graph_nn.py','0ae2f865c8903542c0771ca8a750cf6ca7f01dc2'),
    'multi_task_graph_router.py': ('model/multi_task_graph_router.py','d1ac7048598f950adfea0ade816a97abdcc63864'),
    'config.yaml': ('configs/config.yaml','36e4b54b6f9ad6e3586d9e45fca5a72157398edd'),
    'requirements.txt': ('requirements.txt','87a7aa973e823d6f456a16d89c9a116b767c1523'),
}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def upstream_sources(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sources={};records={}
    for name,(remote,expected) in SOURCE_BLOBS.items():
        path=root/name
        if not path.exists():
            url=f'https://raw.githubusercontent.com/ulab-uiuc/GraphRouter/{UPSTREAM}/{remote}'
            with urllib.request.urlopen(url,timeout=30) as response:path.write_bytes(response.read())
        raw=path.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if actual!=expected:raise ValueError(f'upstream checksum mismatch: {name}')
        sources[name]=raw.decode();records[name]=dict(git_blob=actual,sha256=hashlib.sha256(raw).hexdigest())
    return sources,records


def ast_facts(source,local_source):
    def classes(s):
        return {n.name:n for n in ast.parse(s).body if isinstance(n,ast.ClassDef)}
    a,b=classes(source),classes(local_source)
    same={name:ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
          for name in ('FeatureAlign','EncoderDecoderNet')}
    methods={n.name:n for n in a['GNN_prediction'].body if isinstance(n,ast.FunctionDef)}
    assignments=[n for n in ast.walk(methods['train_validate']) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='best_f1' for t in n.targets)]
    test_calls=[n for n in ast.walk(methods['test']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    random_calls=[n for n in ast.walk(methods['train_validate']) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='rand'
                  and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch']
    default_mask_device=bool(random_calls) and all(not any(k.arg=='device' for k in n.keywords) for n in random_calls)
    return dict(core_classes_identical=same,
                upstream_best_f1_assignment_count=len(assignments),
                upstream_best_f1_never_updated=len(assignments)==1,
                upstream_test_reloads_checkpoint=any(n.func.attr=='load_state_dict' for n in test_calls),
                upstream_masks_cpu_default=default_mask_device,
                note=('best_f1 is updated and best-validation state is saved; public test() evaluates current state'
                      if len(assignments)>1 else 'best_f1 is not updated after initialization'))


def parity_fixture():
    torch.manual_seed(731)
    tasks=[t for t in ('alpaca_data','GSM8K','multi_news','SQUAD') for _ in range(10)]
    return dict(query=torch.randn(40,5),task=torch.randn(40,6),model=torch.randn(10,6),
                task_names=tasks,performance=torch.rand(40,10),cost=torch.rand(40,10),
                splits=partitions(tasks,support_size=8))


def execute_upstream_parity(source):
    """Execute the ACTUAL pinned public loop for one fixture epoch, not a rewrite."""
    ns={}
    exec(compile(source,'pinned_upstream_graph_nn.py','exec'),ns)
    d=parity_fixture();n=len(d['query']);k=6;sp=d['splits']
    reward=.5*d['performance'][:,:k]-.5*d['cost'][:,:k]
    label=F.one_hot(reward.argmax(1),num_classes=k).reshape(-1,1).float()
    masks={}
    for key in ('train','validation','test'):
        mask=torch.zeros(n,k);mask[sp[key]]=1;masks[key]=mask.flatten()
    with tempfile.TemporaryDirectory() as tmp:
        config=dict(train_epoch=1,batch_size=3,train_mask_rate=.5,llm_num=6,
                    model_path=str(Path(tmp)/'upstream.pth'),learning_rate=.0003,weight_decay=.0001)
        logger=types.SimpleNamespace(log=lambda x:None)
        torch.manual_seed(19)
        original=ns['GNN_prediction'](5,6,8,3,logger,config,'cpu')
        ours=EncoderDecoderNet(5,6,8,3)
        ours.load_state_dict(copy.deepcopy(original.model.state_dict()))
        opt=torch.optim.AdamW(ours.parameters(),lr=.0003,weight_decay=.0001)
        form=ns['form_data']('cpu')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            formed={key:form.formulation(task_id=d['task'],query_feature=d['query'],llm_feature=d['model'][:6],
                org_node=[q for q in range(n) for _ in range(k)],
                des_node=list(range(k))*n,edge_feature=reward.flatten(),label=label,
                edge_mask=masks[key],combined_edge=torch.stack((d['cost'][:,:k],d['performance'][:,:k]),-1),
                train_mask=masks['train'],valide_mask=masks['validation'],test_mask=masks['test'])
                for key in masks}
        calls_old=[];calls_new=[]
        def hook_old(model,args,kwargs,out):
            calls_old.append(dict(training=model.training,prediction_mask=kwargs['edge_mask'].clone(),
                                  visible_mask=kwargs['edge_can_see'].clone(),scores=out.detach().clone()))
        def hook_new(model,args,kwargs,out):
            calls_new.append(dict(training=model.training,prediction_mask=kwargs['edge_mask'].clone(),
                                  visible_mask=kwargs['edge_can_see'].clone(),scores=out.detach().clone()))
        ho=original.model.register_forward_hook(hook_old,with_kwargs=True)
        hn=ours.register_forward_hook(hook_new,with_kwargs=True)
        torch.manual_seed(113)
        with contextlib.redirect_stdout(io.StringIO()),warnings.catch_warnings():
            warnings.simplefilter('ignore')
            original.train_validate(formed['train'],formed['validation'],formed['test'])
        torch.manual_seed(113)
        train_step(ours,opt,training_graph(d,'cpu'),draws=3)
        train_old=[x for x in calls_old if x['training']]
        train_new=[x for x in calls_new if x['training']]
        masks_equal=all(torch.equal(a['prediction_mask'],b['prediction_mask']) and
                        torch.equal(a['visible_mask'],b['visible_mask']) for a,b in zip(train_old,train_new))
        max_score=max(float((a['scores']-b['scores']).abs().max()) for a,b in zip(train_old,train_new))
        state_delta=max(float((v.float()-ours.state_dict()[key].float()).abs().max())
                        for key,v in original.model.state_dict().items())
        ho.remove();hn.remove()
        # Compare evaluation using original form_data graph and reconstructed graph.
        from router.graphrouter_reference import predict
        ours.eval();known=tuple(range(6))
        own_val=predict(ours,d,known,sp['train'],[known]*len(sp['train']),sp['validation'],'cpu')
        ctx=sp['train']+sp['validation']
        own_test=predict(ours,d,known,ctx,[known]*len(ctx),sp['test'],'cpu')
        evaluations=[x for x in calls_old if not x['training']]
        val_delta=float((evaluations[0]['scores'].reshape_as(own_val)-own_val).abs().max())
        test_delta=float((evaluations[1]['scores'].reshape_as(own_test)-own_test).abs().max())
        source_order_test=predict(ours,d,known,sorted(ctx),[known]*len(ctx),sp['test'],'cpu')
        source_order_delta=float((evaluations[1]['scores'].reshape_as(source_order_test)-source_order_test).abs().max())
    passed=(len(train_old)==len(train_new)==3 and masks_equal and
            max(max_score,state_delta,val_delta,test_delta)<1e-6)
    return dict(passed=passed,fixture_only=True,benchmark_training_repeated=False,
                executed_function='actual pinned GNN_prediction.train_validate',
                mask_draws=3,epochs=1,masks_equal=masks_equal,max_training_score_error=max_score,
                max_parameter_or_bn_error=state_delta,max_validation_score_error=val_delta,max_test_score_error=test_delta,
                max_test_score_error_with_upstream_edge_order=source_order_delta,
                numerical_note='Reference concatenates train+val; upstream masks full edges in source order. Sum-order rounding is reported separately.')


def canonical_checks(result,data):
    sp=data['splits']
    if result['splits']!=sp:raise ValueError('canonical splits do not match released CSV preparation')
    if result['provenance']['source_csv_sha256']!=RELEASE_CSV_SHA256:raise ValueError('canonical source mismatch')
    targets=sp['test'];models=result['model_names']
    max_error=0.
    for condition in ('known_only','description_only','fewshot_80','fixed_known_validation'):
        summary=result['results'][condition];records=summary['selections']
        if [x['query_index'] for x in records]!=targets:raise ValueError('canonical target order mismatch')
        for row in records:
            qi,mi=row['query_index'],models.index(row['model'])
            p,c=float(data['performance'][qi,mi]),float(data['cost'][qi,mi])
            max_error=max(max_error,abs(row['P']-p),abs(row['C']-c),abs(row['R']-(.5*p-.5*c)))
        for key in ('P','C','R'):
            max_error=max(max_error,abs(summary[key]-sum(x[key] for x in records)/len(records)))
    scores=result['candidate_scores']
    score_choices={}
    for condition in ('known_only','description_only','fewshot_80'):
        sc=scores[condition]
        chosen=[max(range(len(row)),key=lambda j:row[j]) for row in sc]
        claimed=[models.index(x['model']) for x in result['results'][condition]['selections']]
        score_choices[condition]=chosen==claimed
    support_changed=max(abs(a[j]-b[j]) for a,b in zip(scores['description_only'],scores['fewshot_80']) for j in range(6,10))
    return dict(passed=max_error<1e-9 and all(score_choices.values()),maximum_metric_error=max_error,
                score_choices_match=score_choices,support_changes_unseen_scores=support_changed>0,
                max_support_score_change=support_changed,queries=len(targets),epochs=len(result['history']),
                fewshot_R=result['results']['fewshot_80']['R'],new_model_selections=sum(
                    n for m,n in result['results']['fewshot_80']['model_counts'].items() if m in MODELS[6:]),
                useful_incremental_gain=result['results']['fewshot_80']['vs_known']['gain'],
                public_rounded_reward_match=round(result['results']['fewshot_80']['R'],3)==.207)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream-dir',type=Path,default=Path('data/graphrouter_fidelity/upstream'))
    ap.add_argument('--prepared-path',type=Path,default=Path('data/graphrouter_reference/prepared.pt'))
    ap.add_argument('--results-path',type=Path,default=Path('analysis/graphrouter_reference/seed1881_balance/results.json'))
    ap.add_argument('--output-path',type=Path,default=Path('outputs/graphrouter_reference/fidelity_audit.json'))
    args=ap.parse_args()
    if args.output_path.exists():raise FileExistsError(f'refusing to overwrite {args.output_path}')
    sources,hashes=upstream_sources(args.upstream_dir)
    facts=ast_facts(sources['graph_nn.py'],Path('router/graphrouter_official.py').read_text())
    step=execute_upstream_parity(sources['graph_nn.py'])
    data=torch.load(args.prepared_path,map_location='cpu',weights_only=True)
    canonical=json.loads(args.results_path.read_text());metrics=canonical_checks(canonical,data)
    if not all(facts['core_classes_identical'].values()) or not step['passed'] or not metrics['passed']:
        raise RuntimeError('concrete fidelity discrepancy: inspect source/step/metric evidence before any benchmark rerun')
    report=dict(verdict='partial_reproduction',scope='pinned public release; not a full reconstruction of unpublished Table5 execution',
                upstream_commit=UPSTREAM,source_files=hashes,static_evidence=facts,
                audit_runtime=dict(torch=torch.__version__,pyg=torch_geometric.__version__,sklearn=sklearn.__version__,device='cpu'),
                executable_parity=step,canonical_result_checks=metrics,
                source_results_sha256=file_sha(args.results_path),source_csv_sha256=RELEASE_CSV_SHA256,
                corrections=[dict(kind='prospective_GPU_rng',issue='mask random draws must remain on CPU as upstream, not CUDA',
                                  affects_recorded_cpu_primary=False,benchmark_rerun_required=False)],
                missing_evidence=['Table5 auxiliary query IDs and exact sampler state',
                                  'few-shot driver: whether unseen nodes participate during training and exact history construction',
                                  'Table5 checkpoint-to-configuration mapping and per-query predictions',
                                  'exact PyG version; public requirements pins Torch2.1.0 while measured run uses2.15.0',
                                  'resolution of GAT32/Adam schedule prose versus GeneralConv16/AdamW code and Cases unit'],
                full_benchmark_retrained=False,promorouter_modified=False,
                rejected_manual_suspicion='An initial reading suggested best_f1 was not updated; AST and actual source show it is updated. No checkpoint-saving defect was found.',
                decision='No error found that changes the recorded CPU primary. No new fit or parameter search. Full useful-adaptation replication remains unverified.')
    args.output_path.parent.mkdir(parents=True,exist_ok=True)
    args.output_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

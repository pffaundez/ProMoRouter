import copy
import csv
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_reference import (MODELS, NumpyArrayUnpickler, embedding,
    graph, partitions, predict, read_public_csv)
from train_graphrouter_reference import run, training_graph, train_step


def synthetic():
    torch.manual_seed(44)
    n=40
    return dict(query=torch.randn(n,5),task=torch.randn(n,6),model=torch.randn(10,6),
                task_names=[t for t in ('alpaca_data','GSM8K','multi_news','SQUAD') for _ in range(10)],
                performance=torch.rand(n,10),cost=torch.rand(n,10),
                splits=partitions([t for t in ('alpaca_data','GSM8K','multi_news','SQUAD') for _ in range(10)],support_size=8),
                metadata=dict(mode='synthetic_engineering',paper_population_match=False,model_names=list(MODELS)))


class GraphRouterReferenceTests(unittest.TestCase):
    def test_paper_splits_six_four_and_support_only_from_train(self):
        names=[t for t in ('alpaca_data','GSM8K','multi_news','SQUAD') for _ in range(600)]
        p=partitions(names)
        self.assertEqual([len(p[k]) for k in ('train','validation','test','support')],[1680,240,480,80])
        self.assertTrue(set(p['support'])<=set(p['train']))
        self.assertFalse(set(p['support']) & (set(p['validation'])|set(p['test'])))
        self.assertEqual(p,partitions(names))

    def test_unexpected_task_layout_fails(self):
        with self.assertRaises(ValueError):partitions(['a']*10+['b']*11,support_size=2)
        with self.assertRaises(ValueError):partitions(['a','b']*10,support_size=2)

    def test_numpy_embedding_formats_and_restricted_pickle(self):
        self.assertTrue(torch.equal(embedding('[[1.0, 2.0, -3e-2]]'),embedding('[[ 1.0\n 2.0 -3e-2 ]]')))
        with self.assertRaises(ValueError):embedding('[[ nan ]]')
        with self.assertRaises(pickle.UnpicklingError):NumpyArrayUnpickler(io.BytesIO(pickle.dumps(Path('x')))).load()
        x=np.arange(12,dtype=np.float32).reshape(3,4)
        self.assertTrue(np.array_equal(x,NumpyArrayUnpickler(io.BytesIO(pickle.dumps(x))).load()))

    def test_public_csv_semantic_alignment_and_no_renormalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'router_data.csv'
            fields=['task_id','query','query_embedding','task_description_embedding','llm','effect','cost']
            rows=[dict(task_id='a',query='q',query_embedding='[[1. 2.]]',
                       task_description_embedding='[[3. 4.]]',llm=m,effect=.2+i*.01,cost=.8)
                  for i,m in enumerate(MODELS)]
            def write():
                with p.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            write();d=read_public_csv(p)
            self.assertEqual(d['query'].shape,(1,2));self.assertAlmostEqual(float(d['cost'][0,0]),.8)
            rows[3]['llm']='';write()
            with self.assertRaises(ValueError):read_public_csv(p)
            resolved=read_public_csv(p,allow_upstream_blank_label=True)
            self.assertEqual(resolved['positional_blank_model_labels_resolved'],1)
            rows[3]['llm']=MODELS[3]
            rows[0],rows[1]=rows[1],rows[0];write()
            with self.assertRaises(ValueError):read_public_csv(p)

    def test_training_masks_hidden_outcomes_but_preserves_all_query_features(self):
        d=synthetic();before=training_graph(d,'cpu')
        self.assertEqual(len(before[1]),40)
        self.assertEqual(before[3].shape,(2,240))
        self.assertEqual(int(before[-1].sum()),len(d['splits']['train']))
        d['performance'][:,6:]=float('nan');d['cost'][:,6:]=float('nan')
        for key in ('validation','test'):
            d['performance'][d['splits'][key],:6]=float('nan');d['cost'][d['splits'][key],:6]=float('nan')
        after=training_graph(d,'cpu')
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(before,after)))

    def test_training_step_matches_public_accumulated_mask_loop(self):
        d=synthetic();inputs=training_graph(d,'cpu')
        torch.manual_seed(19);a=EncoderDecoderNet(5,6,8,3);b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=.0003,weight_decay=.0001)
        ob=torch.optim.AdamW(b.parameters(),lr=.0003,weight_decay=.0001)
        torch.manual_seed(24);observed=train_step(a,oa,inputs,draws=3)
        t,q,m,e,attrs,eligible,labels=inputs
        torch.manual_seed(24);b.train();total=0
        for _ in range(3):
            mask=eligible.clone();rm=torch.rand(mask.size())<.5;mask=torch.where(mask & rm,False,mask)
            see=~mask & eligible;ob.zero_grad()
            pred=b(t,q,m,e,edge_mask=mask,edge_can_see=see,edge_weight=attrs)
            total=total+F.binary_cross_entropy(pred.flatten(),labels[mask])
        loss=total/3;loss.backward();ob.step()
        self.assertAlmostEqual(observed,float(loss.detach()),places=6)
        self.assertTrue(all(torch.allclose(v,b.state_dict()[k],atol=1e-7) for k,v in a.state_dict().items()))

    def test_support_changes_only_unseen_scores_and_freezes_bn(self):
        d=synthetic();known=tuple(range(6));full=tuple(range(10));sp=d['splits']
        net=EncoderDecoderNet(5,6,8,3).eval();state={k:v.clone() for k,v in net.state_dict().items()}
        ctx=sp['train']+sp['validation'];models=[known]*len(ctx)
        before=predict(net,d,full,ctx,models,sp['test'],'cpu')
        after=predict(net,d,full,ctx+sp['support'],models+[tuple(range(6,10))]*8,sp['test'],'cpu')
        self.assertTrue(torch.allclose(before[:,:6],after[:,:6],atol=1e-7))
        self.assertFalse(torch.allclose(before[:,6:],after[:,6:]))
        self.assertTrue(all(torch.equal(v,net.state_dict()[k]) for k,v in state.items()))
        d['performance'][sp['test']]=float('nan')
        repeated=predict(net,d,full,ctx,models,sp['test'],'cpu')
        self.assertTrue(torch.equal(before,repeated))

    def test_prediction_target_batch_and_model_order(self):
        d=synthetic();ctx=d['splits']['train'];known=tuple(range(6));targets=d['splits']['test']
        net=EncoderDecoderNet(5,6,8,3).eval()
        a=predict(net,d,known,ctx,[known]*len(ctx),targets,'cpu')
        b=predict(net,d,tuple(reversed(known)),ctx,[known]*len(ctx),list(reversed(targets)),'cpu')
        self.assertTrue(torch.allclose(a,b.flip((0,1)),atol=1e-7))
        c=predict(net,d,known,ctx,[known]*len(ctx),targets[:1],'cpu')
        self.assertTrue(torch.allclose(a[:1],c,atol=1e-7))

    def test_visible_target_overlap_rejected(self):
        d=synthetic()
        with self.assertRaises(ValueError):graph(d,tuple(range(6)),[0],[(0,)], [0],'cpu')

    def test_two_epoch_reference_runs_three_frozen_conditions(self):
        d=synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            r=run(d,2,'cpu',out)
            self.assertEqual(r['model_names'],list(MODELS))
            self.assertEqual(r['training']['internal_width'],16)
            self.assertFalse(r['publication_reference']['quantitative_reproduction_verified'])
            self.assertEqual(r['calibration_calls'],32) # Synthetic fixture has 8 support qids.
            self.assertTrue(r['adaptation_frozen_state_verified'])
            self.assertTrue((out/'router.pt').exists())
            for name in ('known_only','description_only','fewshot_80'):
                v=r['results'][name];self.assertAlmostEqual(v['R'],.5*v['P']-.5*v['C'])
            with self.assertRaises(FileExistsError):run(d,1,'cpu',out)


if __name__=='__main__':unittest.main()

import unittest

import torch
from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_joint import (action_data, best_action_labels, cost_reference,
    graph_inputs, make_visible, evaluate_scores)
from train_graphrouter_unseen import evaluate_scores as direct_scores
from train_graphrouter_unseen import make_visible as direct_visible
from train_graphrouter_unseen import direct_data
from train_router_support_graph import PROMPTS


class JointGraphRouterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.pool = ('mistral-7b','qwen2.5-7b','qwen2.5-14b')
        self.seen = self.pool[:2]
        self.rows = [dict(qid='s',task='gsm8k',action_edges=[
            dict(prompt=p,model=m,performance=.2+.1*mi+.04*pi,
                 cost_proxy_money=1.+mi+.3*pi,cost_norm_query=(mi+pi)/6)
            for pi,p in enumerate(PROMPTS) for mi,m in enumerate(self.pool)])]
        self.targets = [dict(qid='t',task='gsm8k'),dict(qid='u',task='gsm8k')]
        self.tables = ({q:torch.randn(5) for q in ('s','t','u')},
                       {'gsm8k':torch.randn(6)}, {p:torch.randn(4) for p in PROMPTS},
                       {m:torch.randn(6) for m in self.pool})
        self.net = EncoderDecoderNet(9,6,8,3).eval()
        self.values = action_data(self.rows,PROMPTS,self.pool,1.,2.)
        self.visible = make_visible(self.rows,PROMPTS,self.seen,self.values[:,:,:2],.9)

    def score(self, pool=None, visible=None, targets=None, prompts=PROMPTS, net=None):
        return evaluate_scores(net or self.net,self.rows,self.targets if targets is None else targets,
                               prompts,pool or self.pool,self.tables,
                               self.visible if visible is None else visible,'cpu')

    def test_graph_counts_and_single_shared_node_per_model(self):
        task, query, models, src, dst, attrs = graph_inputs(self.rows,self.targets,PROMPTS,self.seen,self.tables,self.visible,'cpu')
        self.assertEqual(query.shape,(12,9))
        self.assertEqual(task.shape,(12,6))
        self.assertEqual(models.shape,(2,6))
        self.assertEqual(len(src),8)
        self.assertEqual(set(dst),{12,13})
        self.assertEqual(set(src),set(range(4)))
        self.assertEqual(len(attrs),8)

    def test_unseen_and_target_outcomes_never_enter_scorer(self):
        before = self.score()
        for e in self.rows[0]['action_edges']:
            if e['model'] == self.pool[-1]:
                e['performance'] = float('nan'); e['cost_proxy_money'] = float('nan')
        self.targets[0]['action_edges'] = [dict(performance=float('nan'))]
        # Rebuild known context to check that excluded outcomes are never read.
        values = action_data(self.rows,PROMPTS,self.seen,1.,2.)
        visible = make_visible(self.rows,PROMPTS,self.seen,values,.9)
        self.assertTrue(torch.equal(before,self.score(visible=visible)))

    def test_calibration_sensitivity_known_invariance_and_frozen_buffers(self):
        state = {k:v.clone() for k,v in self.net.state_dict().items()}
        before = self.score()
        extra = make_visible(self.rows,PROMPTS,self.pool[2:],self.values[:,:,2:],.9)
        after = self.score(visible=self.visible+extra)
        self.assertTrue(torch.allclose(before[:,:,:2],after[:,:,:2],atol=1e-7))
        self.assertFalse(torch.allclose(before[:,:,2],after[:,:,2],atol=1e-7))
        self.assertTrue(all(torch.equal(v,self.net.state_dict()[k]) for k,v in state.items()))
        seen_only = self.score(pool=self.seen)
        self.assertTrue(torch.allclose(before[:,:,:2],seen_only,atol=1e-7))

    def test_target_batch_isolation_and_order_equivariance(self):
        before = self.score()
        alone = self.score(targets=self.targets[1:])
        self.assertTrue(torch.allclose(alone,before[1:],atol=1e-7))
        reverse = self.score(pool=tuple(reversed(self.pool)),prompts=tuple(reversed(PROMPTS)))
        self.assertTrue(torch.allclose(reverse.flip((1,2)),before,atol=1e-7))
        target_reverse = self.score(targets=list(reversed(self.targets)))
        self.assertTrue(torch.allclose(target_reverse.flip(0),before,atol=1e-7))

    def test_prompt_identity_changes_source_not_destination_model(self):
        before = self.score()
        self.tables[2]['direct'] += 10.
        after = self.score()
        self.assertFalse(torch.allclose(before[:,0],after[:,0]))
        # A changed historical prompt can update shared model states; other
        # prompts' scores need not remain fixed. Model IDs must stay aligned.
        _,_,models,_,dst,_ = graph_inputs(self.rows,self.targets,PROMPTS,self.pool,self.tables,self.visible,'cpu')
        self.assertTrue(torch.equal(models[2],self.tables[3][self.pool[2]]))
        self.assertEqual(len(set(dst)),2)  # Only the two observed models.

    def test_direct_reduction_with_copied_weights_and_zero_prompt_block(self):
        direct = EncoderDecoderNet(5,6,8,3).eval()
        joint = EncoderDecoderNet(9,6,8,3).eval()
        state = {k:v.clone() for k,v in direct.state_dict().items()}
        weight = state['model_align.query_transform.weight']
        state['model_align.query_transform.weight'] = torch.cat((weight,torch.zeros(8,4)),1)
        joint.load_state_dict(state)
        values = direct_data(self.rows,self.seen,1.,2.)
        visible = direct_visible(self.rows,self.seen,values,.9)
        qt,tt,_,mt = self.tables
        old = direct_scores(direct,self.rows,self.targets,self.pool,(qt,tt,mt),visible,'cpu')
        jvisible = make_visible(self.rows,('direct',),self.seen,self.values[:,:1,:2],.9)
        new = self.score(net=joint,prompts=('direct',),visible=jvisible)[:,0]
        self.assertTrue(torch.allclose(old,new,atol=1e-7))

    def test_one_positive_per_original_query_and_extrapolated_cost(self):
        v = self.values.repeat(2,1,1,1)
        best, labels = best_action_labels(v,.9)
        self.assertEqual(labels.reshape(2,-1).sum(1).tolist(),[1.,1.])
        self.assertTrue(torch.equal(best,(v[...,0]-.9*v[...,1]).flatten(1).argmax(1)))
        self.assertGreater(float(v[...,1].max()),1.)
        seen_ref = cost_reference(self.rows,self.seen)
        for e in self.rows[0]['action_edges']:
            if e['model']==self.pool[-1] or e['prompt']!='direct':
                e['cost_proxy_money'] = 1e9
        self.assertEqual(cost_reference(self.rows,self.seen),seen_ref)

    def test_forbidden_and_duplicate_visible_edges_fail(self):
        with self.assertRaises(ValueError):
            self.score(visible=self.visible+[('t','direct',self.seen[0],[1.,0.,1.])])
        with self.assertRaises(ValueError):
            self.score(visible=self.visible+self.visible[:1])
        with self.assertRaises(ValueError):
            self.score(targets=self.rows)

    def test_numeric_outcomes_and_reference_fail_loudly(self):
        with self.assertRaises(ValueError):
            action_data(self.rows,PROMPTS,self.seen,1.,0.)
        del self.rows[0]['action_edges'][0]['cost_proxy_money']
        with self.assertRaises(KeyError):
            action_data(self.rows,PROMPTS,self.seen,1.,2.)


if __name__ == '__main__':
    unittest.main()

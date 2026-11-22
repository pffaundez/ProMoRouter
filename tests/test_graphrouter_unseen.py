import unittest
import torch
from router.graphrouter_official import EncoderDecoderNet
from train_graphrouter_unseen import (
    direct_data, nested_support, macro_f1, make_visible, evaluate_scores,
)


class GraphRouterControlTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.rows=[dict(qid='s',task='gsm8k',action_edges=[
            dict(prompt='direct',model=m,performance=p,cost_proxy_money=c,cost_norm_query=.2)
            for m,p,c in (('mistral-7b',.4,1.),('qwen2.5-7b',.7,2.),('qwen2.5-14b',.9,3.))])]
        self.targets=[dict(qid='t',task='gsm8k'),dict(qid='u',task='gsm8k')]
        self.pool=('mistral-7b','qwen2.5-7b','qwen2.5-14b')
        self.tables=({'s':torch.randn(5),'t':torch.randn(5),'u':torch.randn(5)},
                     {'gsm8k':torch.randn(6)}, {m:torch.randn(6) for m in self.pool})
        self.net=EncoderDecoderNet(5,6,8,3).eval()
        self.values=direct_data(self.rows,self.pool,1.,2.)

    def test_nested_support_preserves_eight_and_no_evaluation_queries(self):
        small,large=nested_support(set(map(str,range(100))),8,80,1)
        self.assertEqual((len(small),len(large)),(8,80))
        self.assertTrue(small<=large)
        self.assertEqual((small,large),nested_support(set(map(str,range(100))),8,80,1))

    def test_masked_model_and_target_outcomes_do_not_enter_context(self):
        seen=self.pool[:2]
        data=direct_data(self.rows,seen,1.,2.)
        visible=make_visible(self.rows,seen,data,.9)
        before=evaluate_scores(self.net,self.rows,self.targets,self.pool,self.tables,visible,'cpu')
        # The absent held-out outcome may be arbitrary, including NaN.
        self.rows[0]['action_edges'][-1]['performance']=float('nan')
        self.targets[0]['action_edges']=[dict(performance=float('nan'))]
        after=evaluate_scores(self.net,self.rows,self.targets,self.pool,self.tables,visible,'cpu')
        self.assertTrue(torch.equal(before,after))

    def test_calibration_changes_only_new_model_and_frozen_state(self):
        seen=self.pool[:2]
        visible=make_visible(self.rows,seen,self.values[:,:2],.9)
        state={k:v.clone() for k,v in self.net.state_dict().items()}
        before=evaluate_scores(self.net,self.rows,self.targets,self.pool,self.tables,visible,'cpu')
        extra=make_visible(self.rows,self.pool[2:],self.values[:,2:],.9)
        after=evaluate_scores(self.net,self.rows,self.targets,self.pool,self.tables,visible+extra,'cpu')
        self.assertTrue(torch.allclose(before[:,:2],after[:,:2],atol=1e-7))
        self.assertFalse(torch.allclose(before[:,2],after[:,2]))
        self.assertTrue(all(torch.equal(v,self.net.state_dict()[k]) for k,v in state.items()))
        alone=evaluate_scores(self.net,self.rows,self.targets[1:],self.pool,self.tables,visible+extra,'cpu')
        self.assertTrue(torch.allclose(alone,after[1:],atol=1e-7))

    def test_macro_f1_and_seen_only_data(self):
        self.assertAlmostEqual(macro_f1(torch.tensor([0,0,1]),torch.tensor([0,1,1]),3),2/3)
        seen=direct_data(self.rows,self.pool[:2],1.,2.)
        self.rows[0]['action_edges'][2]['cost_proxy_money']=999.
        self.assertTrue(torch.equal(seen,direct_data(self.rows,self.pool[:2],1.,2.)))


if __name__=='__main__':unittest.main()

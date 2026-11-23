from collections import Counter
import unittest

import torch
from router.episodic_incorporation import EpisodeSampler
from router.graphrouter_official import EncoderDecoderNet


class EpisodeTests(unittest.TestCase):
    def sampler(self):return EpisodeSampler(40,range(8,40),4,8,seed=1,support_size=8)

    def test_balanced_roles_and_deterministic_shared_plan(self):
        a,b=self.sampler(),self.sampler();roles=Counter()
        for _ in range(32):
            x,y=a.sample(),b.sample();roles[x['pseudo_index']]+=1
            for key in ('predict','control_visible','episodic_visible','target_queries'):
                self.assertTrue(torch.equal(x[key],y[key]))
            self.assertEqual(x['support_indices'],y['support_indices'])
        self.assertEqual(set(roles.values()),{4})
        self.assertEqual(a.plan_sha256,b.plan_sha256)

    def test_target_queries_have_no_visible_outcomes_in_either_arm(self):
        e=self.sampler().sample();q=e['target_queries']
        self.assertFalse(bool(q[:8].any()))
        self.assertTrue(bool(q.any()))
        self.assertTrue(bool(e['predict'].reshape(40,4,8)[q].all()))
        for key in ('control_visible','episodic_visible'):
            self.assertFalse(bool(e[key].reshape(40,4,8)[q].any()))
        self.assertFalse(any(bool(q[i]) for i in e['support_indices']))

    def test_only_pseudo_history_changes_and_budget_is_exact(self):
        e=self.sampler().sample();pseudo=e['pseudo_index']
        control=e['control_visible'].reshape(40,4,8);epi=e['episodic_visible'].reshape(40,4,8)
        self.assertEqual(int(epi[:,:,pseudo].sum()),32)
        for mi in range(8):
            if mi!=pseudo:self.assertTrue(torch.equal(control[:,:,mi],epi[:,:,mi]))
        observed_q=epi[:,:,pseudo].any(1).nonzero().flatten().tolist()
        self.assertEqual(observed_q,e['support_indices'])

    def test_hidden_targets_and_non_support_pseudo_outcomes_cannot_change_scores(self):
        torch.manual_seed(7);e=self.sampler().sample();nq=40*4
        task=torch.randn(nq,6);query=torch.randn(nq,9);models=torch.randn(8,6)
        sources=torch.arange(nq).repeat_interleave(8)
        destinations=(torch.arange(8)+nq).repeat(nq)
        edges=torch.stack((sources,destinations));attrs=torch.randn(len(sources),3)
        net=EncoderDecoderNet(9,6,8,3).eval();mask=e['predict'];see=e['episodic_visible']
        with torch.inference_mode():
            before=net(task,query,models,edges,edge_mask=mask,edge_can_see=see,edge_weight=attrs)
            poisoned=attrs.clone();poisoned[~see]=float('nan')
            after=net(task,query,models,edges,edge_mask=mask,edge_can_see=see,edge_weight=poisoned)
        self.assertTrue(torch.equal(before,after))

    def test_invalid_population_rejected(self):
        with self.assertRaises(ValueError):EpisodeSampler(40,range(40),4,8)
        with self.assertRaises(ValueError):EpisodeSampler(40,[1,1],4,8)


if __name__=='__main__':unittest.main()

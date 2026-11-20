import unittest
import torch
from audit_support_graph import score_diagnostics, fixed_task_actions


class AuditTest(unittest.TestCase):
    def test_ranks_margins_and_opportunity_subset(self):
        before=torch.zeros(2,4,2)
        before[:,:,0]=2
        after=before.clone();after[:,:,1]=1
        after[0,0,1]=3
        truth=torch.zeros(2,4,2,2)
        truth[0,0,1,0]=1
        result=score_diagnostics(before,after,truth,1,.9,['a','b'])
        selected=result['summary']['oracle_opportunities']
        self.assertEqual(selected['queries'],1)
        self.assertEqual(selected['before']['median_best_pair_rank'],5)
        self.assertEqual(selected['after']['median_best_pair_rank'],1)
        self.assertEqual(selected['after']['mean_margin_to_seen'],1)
        self.assertEqual(result['max_abs_seen_score_delta'],0)

    def test_fixed_actions_selected_only_from_validation(self):
        validation=[{'task':'gsm8k'},{'task':'gsm8k'}]
        truth=torch.zeros(2,4,2,2)
        truth[:,2,1,0]=1
        chosen=fixed_task_actions(validation,truth,['a','b'],.9)
        self.assertEqual(chosen['gsm8k']['prompt'],'decompose')
        self.assertEqual(chosen['gsm8k']['model'],'b')
        self.assertFalse(chosen['gsm8k']['fallback_global'])
        self.assertTrue(chosen['squad']['fallback_global'])


if __name__=='__main__':unittest.main()

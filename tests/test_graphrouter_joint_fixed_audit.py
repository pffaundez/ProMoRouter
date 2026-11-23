import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import unittest
from audit_graphrouter_joint_fixed import audit, choose_policies, fixed_metrics, main


class FixedAuditTests(unittest.TestCase):
    def setUp(self):
        self.prompts = ['direct','decompose']
        self.seen = ['a','b']
        def row(q,kind):
            edges=[]
            for p in self.prompts:
                for m in ['a','b','c']:
                    v = .1
                    if (p,m)==('decompose','a'): v=.8 if kind=='val' else .3
                    if (p,m)==('direct','c'): v=.9 if kind=='support' else .2
                    edges.append(dict(prompt=p,model=m,performance=v,cost_proxy_money=0.))
            return dict(qid=q,task='t',action_edges=edges)
        self.rows = [row('train','train'),row('s','support'),row('v','val'),row('z','test'),row('x','test')]
        truth = [[[[e['performance'],0.,0.] for e in row['action_edges'][i:i+3]]
                  for i in (0,3)] for row in self.rows[-2:]]
        self.result = dict(protocol='official_graphrouter_joint_unseen_v1',
            config=dict(seed=1,reward_weight=.9,held_out_model='c'),
            cost_reference=dict(min=0.,max=1.),prompt_order=self.prompts,
            model_order=['a','b','c'],seen_models=self.seen,
            train_qids=['train'],support_qids=['s'],validation_qids=['v'],test_qids=['x','z'],
            target_outcomes=truth,results=dict(calibrated_8=dict(
                pair_counts={'direct|c':2},selections=[dict(qid=q,R=.2) for q in ('z','x')])))

    def test_selector_never_uses_unseen_validation(self):
        for e in self.rows[2]['action_edges']:
            if e['model']=='c':e['performance']=float('nan')
        policies,evidence=choose_policies([self.rows[2]],[self.rows[1]],self.prompts,self.seen,'c',0.,1.,.9)
        self.assertEqual(policies['fixed_known'],('decompose','a'))
        self.assertEqual(policies['fixed_new_support'],('direct','c'))
        self.assertEqual(policies['fixed_adapted'],('direct','c'))

    def test_adaptation_tie_keeps_known(self):
        for e in self.rows[1]['action_edges']:
            if (e['prompt'],e['model'])==('direct','c'):e['performance']=.3
        p,_=choose_policies([self.rows[2]],[self.rows[1]],self.prompts,self.seen,'c',0.,1.,.9)
        self.assertEqual(p['fixed_adapted'],p['fixed_known'])

    def test_saved_outcome_order_is_selection_order(self):
        self.result['target_outcomes'][0][0][2][0]=.7
        m=fixed_metrics(self.result,('direct','c'))
        self.assertEqual([x['qid'] for x in m['selections']],['z','x'])
        self.assertEqual(m['selections'][0]['R'],.7)

    def test_test_outcomes_cannot_change_policy_choice(self):
        first=audit(self.result,self.rows)
        self.result['target_outcomes'][0][0][2][0]=1.
        second=audit(self.result,self.rows)
        self.assertEqual(first['frozen_policies_sha256'],second['frozen_policies_sha256'])
        self.assertEqual(first['baselines']['fixed_known']['action'],second['baselines']['fixed_known']['action'])
        self.assertFalse(second['instance_routing_advantage_demonstrated'])

    def test_partition_and_order_mismatch_rejected(self):
        bad=copy.deepcopy(self.result);bad['validation_qids']=['s']
        with self.assertRaises(ValueError):audit(bad,self.rows)
        with self.assertRaises(ValueError):audit(self.result,self.rows[:-2]+list(reversed(self.rows[-2:])))

    def test_nonwinning_control_closes_without_more_search(self):
        r=audit(self.result,self.rows)
        self.assertEqual(r['verdict'],'no_advantage_over_both_fixed_controls')
        self.assertEqual(r['comparisons']['fixed_adapted']['gain'],0.)
        self.assertIn('No automatic refit',r['closure'])


    def test_cli_writes_new_audit_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data.jsonl';source=root/'results.json'
            data.write_text('\n'.join(map(json.dumps,self.rows)))
            self.result['dataset_sha256']=hashlib.sha256(data.read_bytes()).hexdigest()
            source.write_text(json.dumps(self.result));before=source.read_bytes()
            argv=['audit_graphrouter_joint_fixed.py','--data-path',str(data),'--results-path',str(source)]
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                main()
            report=json.loads((root/'fixed_policy_audit.json').read_text())
            self.assertEqual(report['source_results_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(before,source.read_bytes())
            with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):main()


if __name__=='__main__':unittest.main()

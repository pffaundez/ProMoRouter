import json
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from router.graphrouter_official import EncoderDecoderNet
from router.graphrouter_reference import MODELS,RELEASE_CSV_SHA256
from train_graphrouter_reference import training_graph,train_step,summarize
from audit_graphrouter_fidelity import ast_facts,execute_upstream_parity,parity_fixture,canonical_checks

FIXTURE=Path(__file__).parent/'fixtures/graphrouter_upstream_graph_nn.py'


class FidelityAuditTests(unittest.TestCase):
    def test_actual_upstream_trainer_and_validation_test_parity(self):
        raw=FIXTURE.read_bytes()
        self.assertEqual(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                         '0ae2f865c8903542c0771ca8a750cf6ca7f01dc2')
        report=execute_upstream_parity(FIXTURE.read_text())
        self.assertTrue(report['passed'])
        self.assertTrue(report['masks_equal'])
        self.assertEqual(report['max_parameter_or_bn_error'],0.)
        self.assertLess(report['max_test_score_error'],1e-6)
        self.assertFalse(report['benchmark_training_repeated'])

    def test_ast_distinguishes_saved_best_from_current_test_state(self):
        report=ast_facts(FIXTURE.read_text(),Path('router/graphrouter_official.py').read_text())
        self.assertTrue(all(report['core_classes_identical'].values()))
        self.assertEqual(report['upstream_best_f1_assignment_count'],2)
        self.assertFalse(report['upstream_best_f1_never_updated'])
        self.assertFalse(report['upstream_test_reloads_checkpoint'])

    def test_masks_are_explicitly_sampled_on_cpu(self):
        d=parity_fixture();net=EncoderDecoderNet(5,6,8,3)
        opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
        original=torch.rand
        with patch('train_graphrouter_reference.torch.rand',wraps=original) as draw:
            train_step(net,opt,training_graph(d,'cpu'),draws=1)
        self.assertEqual(draw.call_args.kwargs['device'],'cpu')

    def test_metric_audit_detects_changed_reward(self):
        d=parity_fixture();test=d['splits']['test'];n=len(test)
        known=torch.zeros(n,6);known[:,0]=1.
        desc=torch.zeros(n,10);desc[:,0]=1.
        few=desc.clone();few[:,6:]=.2
        results={name:summarize(d,test,pool,scores) for name,pool,scores in
                 [('known_only',tuple(range(6)),known),('description_only',tuple(range(10)),desc),('fewshot_80',tuple(range(10)),few)]}
        results['fewshot_80']['vs_known']={'gain':0.}
        results['fixed_known_validation']=results['known_only']
        r=dict(splits=d['splits'],provenance={'source_csv_sha256':RELEASE_CSV_SHA256},
               model_names=list(MODELS),results=results,history=[{}],
               candidate_scores={k:v.tolist() for k,v in [('known_only',known),('description_only',desc),('fewshot_80',few)]})
        self.assertTrue(canonical_checks(r,d)['passed'])
        r=json.loads(json.dumps(r));r['results']['fewshot_80']['selections'][0]['R']+=.1
        self.assertFalse(canonical_checks(r,d)['passed'])


if __name__=='__main__':unittest.main()

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

import train_graphrouter_joint_unseen as module
from train_router_support_graph import TASKS


class JointIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_and_three_frozen_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(77)
            rows = [dict(qid=f'q{i:03}',task=TASKS[i%4],action_edges=[
                dict(prompt=p,model=m,performance=((i+pi+mi)%9)/8,
                     cost_proxy_money=1.+mi+pi*.7+(i%3)*.2,
                     cost_norm_query=(mi+pi)/12)
                for pi,p in enumerate(module.PROMPTS) for mi,m in enumerate(module.MODELS)])
                for i in range(40)]
            data = root/'data.jsonl'; data.write_text('\n'.join(map(json.dumps,rows)))
            tables = dict(query={r['qid']:torch.randn(5) for r in rows},
                          task={t:torch.randn(6) for t in TASKS},
                          prompt={p:torch.randn(4) for p in module.PROMPTS},
                          model={m:torch.randn(6) for m in module.MODELS})
            argv = ['train_graphrouter_joint_unseen.py','--data-path',str(data),
                    '--device','cpu','--epochs','2','--split-manifest',str(root/'split.json'),
                    '--direct-results-path',str(root/'missing.json'),'--output-dir',str(root/'output')]
            for name,table in tables.items():
                path = root/f'{name}.pt'; torch.save(table,path)
                argv.extend([f'--{name}-emb-path',str(path)])
            with patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            out = root/'output/qwen2.5-14b/seed1_lam0.9'
            result = json.loads((out/'results.json').read_text())
            self.assertTrue((out/'router.pt').exists())
            self.assertEqual(result['calibration_calls'],32)
            self.assertEqual(result['direct_reference']['status'],'unavailable')
            self.assertEqual(result['input_dimensions']['query_prompt'],9)
            self.assertNotIn('qwen2.5-14b',result['seen_models'])
            self.assertEqual(result['dataset_sha256'],hashlib.sha256(data.read_bytes()).hexdigest())
            groups = [set(result[k]) for k in ('train_qids','support_qids','validation_qids','test_qids')]
            self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))
            self.assertEqual(result['best_epoch'],max(result['history'],key=lambda h:h['validation_macro_f1'])['epoch'])
            for name,v in result['results'].items():
                self.assertAlmostEqual(v['R'],v['P']-.9*v['C'])
                self.assertAlmostEqual(v['R_qnorm'],v['P']-.9*v['C_qnorm'])
                self.assertEqual(len(v['selections']),len(result['test_qids']))
                count = result['graph_counts'][name]
                expected = (len(result['train_qids'])+8)*4*8 + (32 if name=='calibrated_8' else 0)
                self.assertEqual(count['visible_edges'],expected)
                scores = torch.tensor(result['candidate_scores'][name])
                self.assertEqual(tuple(scores.shape),(len(result['test_qids']),4,8 if name=='seen_only' else 9))
            state = torch.load(out/'router.pt',weights_only=True)['state_dict']
            self.assertEqual(state['model_align.query_transform.weight'].shape,(8,9))
            self.assertIn('encoder_conv_1.lin_msg.weight',state)
            # Historical comparison must refuse mismatched metadata, not silently
            # label unlike scales/splits as a matched Direct comparison.
            direct = dict(result)
            direct.update(protocol='official_graphrouter_direct_unseen_v1',small_support_qids=result['support_qids'])
            ref = root/'direct.json'; ref.write_text(json.dumps(direct))
            self.assertTrue(module.compare_direct(ref,result)['numerically_comparable'])
            direct['cost_reference'] = dict(direct['cost_reference'],max=999.)
            ref.write_text(json.dumps(direct))
            self.assertEqual(module.compare_direct(ref,result)['status'],'mismatch')
            with patch.object(sys,'argv',argv), self.assertRaises(FileExistsError):
                module.main()


if __name__ == '__main__':
    unittest.main()

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import train_support_threshold as module


class IntegrationTests(unittest.TestCase):
    def run_experiment(self, passed):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i in range(40):
                rows.append(dict(qid=f'q{i:03}', task='gsm8k', action_edges=[
                    dict(prompt=p, model=m, performance=.4+mi*.01,
                         cost_norm_query=(mi+pi)/12)
                    for pi, p in enumerate(module.PROMPTS)
                    for mi, m in enumerate(module.MODELS)]))
            data = root/'data.jsonl'
            data.write_text('\n'.join(map(json.dumps, rows)))
            tables = dict(query={r['qid']: torch.ones(3) for r in rows},
                          task={t: torch.ones(3) for t in module.TASKS},
                          prompt={p: torch.ones(3) for p in module.PROMPTS},
                          model={m: torch.ones(3) for m in module.MODELS})
            argv = ['train_support_threshold.py', '--data-path', str(data),
                    '--device', 'cpu', '--epochs', '1', '--patience', '1',
                    '--split-manifest', str(root/'split.json'),
                    '--output-dir', str(root/'output')]
            for name, table in tables.items():
                path = root/f'{name}.pt'; torch.save(table, path)
                argv.extend([f'--{name}-emb-path', str(path)])
            calls = []
            original = module.Experiment.evaluate
            def evaluate(exp, net, seen, held, targets):
                calls.append(held)
                if held in module.EXTERNAL:
                    self.assertTrue((root/'output/frozen_threshold.json').exists())
                    self.assertTrue(passed)
                return original(exp, net, seen, held, targets)
            # Force both orchestration paths; this is not a learned positive result.
            with patch.object(sys, 'argv', argv), patch.object(module, 'internal_gate', return_value={'passed': passed}), patch.object(module.Experiment, 'evaluate', evaluate), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            result = json.loads((root/'output/results.json').read_text())
            self.assertEqual(len(calls), 6 if passed else 4)
            self.assertEqual(result['status'], 'completed_external' if passed else 'stopped_internal_gate')
            if passed:
                self.assertEqual(set(result['external_results']), set(module.EXTERNAL))
                self.assertTrue((root/'output/external_shared.pt').exists())
                for r in result['external_results'].values():
                    for name in ('reject_all', 'original_argmax', 'gated'):
                        self.assertAlmostEqual(r[name]['R'], r[name]['P']-.9*r[name]['C'])
            else:
                self.assertFalse((root/'output/external_shared.pt').exists())
                self.assertNotIn('external_results', result)

    def test_stop_before_external(self):
        self.run_experiment(False)

    def test_frozen_external_execution(self):
        self.run_experiment(True)


if __name__ == '__main__':
    unittest.main()

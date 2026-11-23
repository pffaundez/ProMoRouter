import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import train_graphrouter_joint_episodic as module
from train_router_support_graph import TASKS


class PairedIntegrationTests(unittest.TestCase):
    def fixture(self,root):
        torch.manual_seed(55)
        rows=[dict(qid=f'q{i:03}',task=TASKS[i%4],action_edges=[
            dict(prompt=p,model=m,performance=((i+pi+mi)%9)/8,
                 cost_proxy_money=1+mi+.7*pi+.2*(i%3),cost_norm_query=(mi+pi)/12)
            for pi,p in enumerate(module.PROMPTS) for mi,m in enumerate(module.MODELS)]) for i in range(40)]
        data=root/'data.jsonl';data.write_text('\n'.join(map(json.dumps,rows)))
        tables=dict(query={r['qid']:torch.randn(5) for r in rows},task={t:torch.randn(6) for t in TASKS},
                    prompt={p:torch.randn(4) for p in module.PROMPTS},model={m:torch.randn(6) for m in module.MODELS})
        argv=['train_graphrouter_joint_episodic.py','--data-path',str(data),'--device','cpu',
              '--epochs','2','--split-manifest',str(root/'split.json'),'--output-dir',str(root/'run')]
        for name,table in tables.items():
            p=root/f'{name}.pt';torch.save(table,p);argv.extend([f'--{name}-emb-path',str(p)])
        return rows,data,argv

    def test_two_arm_training_freeze_controls_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,data,argv=self.fixture(root);phase={'frozen':False}
            original_fit,original_data=module.fit_pair,module.action_data
            def fit(*args,**kwargs):
                nets,report=original_fit(*args,**kwargs)
                self.assertTrue(all(not p.requires_grad for net in nets.values() for p in net.parameters()))
                phase['frozen']=True
                return nets,report
            def read(rs,prompts,pool,*args):
                if 'qwen2.5-14b' in pool:self.assertTrue(phase['frozen'])
                return original_data(rs,prompts,pool,*args)
            with patch.object(sys,'argv',argv),patch.object(module,'fit_pair',fit),patch.object(module,'action_data',read),contextlib.redirect_stdout(io.StringIO()):
                module.main()
            out=root/'run/qwen2.5-14b/seed1_lam0.9';r=json.loads((out/'results.json').read_text())
            self.assertEqual(set(r['arms']),{'control','episodic'})
            self.assertNotIn('qwen2.5-14b',r['seen_models'])
            self.assertEqual(len(set(r['training']['initial_state_sha256'].values())),1)
            self.assertEqual(r['training']['episode_draws'],64)
            self.assertEqual(set(r['training']['pseudo_model_role_counts'].values()),{8})
            self.assertEqual(r['calibration_actions_per_arm'],32)
            for ex in r['training']['episode_examples']:
                self.assertFalse(set(ex['support_qids']) & set(ex['target_qids']))
                self.assertEqual(ex['observed_actions_per_model'][ex['pseudo_model']],32)
            for name,arm in r['arms'].items():
                self.assertTrue((out/f'{name}.pt').exists())
                for condition,v in arm['results'].items():
                    self.assertAlmostEqual(v['R'],v['P']-.9*v['C'])
                    count=arm['graph_counts'][condition]['visible_edges']
                    self.assertEqual(count,28*4*8+(32 if condition=='calibrated_8' else 0))
                self.assertEqual(len(r['training']['histories'][name]),2)
            a=r['arms']['episodic']['results'];b=r['arms']['control']['results']
            expected=(a['calibrated_8']['R']-a['seen_only']['R'])-(b['calibrated_8']['R']-b['seen_only']['R'])
            self.assertAlmostEqual(r['comparison']['difference_in_incorporation_gain']['gain'],expected)
            with patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):module.main()

    def test_poisoned_real_unseen_outcomes_do_not_change_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,data,argv=self.fixture(root);fitter=module.fit_pair;reports=[]
            def fit(*a,**kw):
                nets,report=fitter(*a,**kw);reports.append(report);return nets,report
            # Stop immediately after both fits so no target/support outcomes are used.
            def stop(*a,**kw):raise RuntimeError('engineering_stop_after_freeze')
            for i in range(2):
                if i:
                    for row in rows:
                        for edge in row['action_edges']:
                            if edge['model']=='qwen2.5-14b':
                                edge['performance']=float('nan');edge['cost_proxy_money']=float('nan')
                    data.write_text('\n'.join(map(json.dumps,rows)))
                    argv[argv.index('--output-dir')+1]=str(root/'poisoned')
                with patch.object(sys,'argv',argv),patch.object(module,'fit_pair',fit),patch.object(module,'evaluate_pair',stop),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'engineering_stop_after_freeze'):module.main()
            self.assertEqual(reports[0]['selected_state_sha256'],reports[1]['selected_state_sha256'])
            self.assertEqual(reports[0]['episode_plan_sha256'],reports[1]['episode_plan_sha256'])

    def test_gpu_request_has_no_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,argv=self.fixture(root);argv[argv.index('--device')+1]='cuda:0'
            with patch.object(sys,'argv',argv),patch.object(module.torch.cuda,'is_available',return_value=False),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):module.main()
            self.assertFalse((root/'run').exists())


if __name__=='__main__':unittest.main()

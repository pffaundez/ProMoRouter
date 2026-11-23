import unittest
import torch
from train_support_threshold import (EXTERNAL, INTERNAL, MODELS, PROMPTS, auc,
    choose_threshold, internal_gate, known_pool, metrics, partition_validation,
    policy_records)


def record(margin, delta, model='llama3.1-8b', qid='q'):
    return dict(qid=qid, margin=margin, delta_R=delta, held_out_model=model,
                P_seen=.2, C_seen=0., P_new=.2+delta, C_new=0.,
                seen_prompt='direct', seen_model='mistral-7b', new_prompt='cot')


class ThresholdTests(unittest.TestCase):
    def test_true_nested_exclusion(self):
        for held in INTERNAL:
            self.assertTrue(set(known_pool(held=held)).isdisjoint((*EXTERNAL, held)))
            self.assertEqual(len(known_pool(held=held)), 6)
        self.assertEqual(len(known_pool()), 7)

    def test_query_check_is_disjoint_and_deterministic(self):
        ids = list(range(119))
        a, b = partition_validation(ids, 1)
        self.assertEqual((len(a), len(b)), (59, 60))
        self.assertFalse(a & b)
        self.assertEqual(a | b, set(ids))
        self.assertEqual((a, b), partition_validation(reversed(ids), 1))

    def test_negative_threshold_rescues_valuable_candidate(self):
        rows = [record(-.4, -.5), record(-.1, .4)]
        t = choose_threshold(rows)
        self.assertEqual(t, -.4)
        self.assertEqual(metrics(rows, 0., .9)['new_selections'], 0)
        self.assertAlmostEqual(metrics(rows, t, .9)['gain'], .2)

    def test_reject_all_wins_nonpositive_and_tied_gains(self):
        self.assertIsNone(choose_threshold([record(-.2, -.1), record(.1, 0.)]))
        self.assertFalse(internal_gate([record(0., 0.)], None, .9)['passed'])
        self.assertIsNone(auc([record(.1, 0.)]))

    def test_gate_uses_independent_check_not_favorable_tuning(self):
        t = choose_threshold([record(-.4, -.5), record(-.1, .4)])
        check = [record(-.4, .4), record(-.1, -.5)]
        self.assertFalse(internal_gate(check, t, .9)['passed'])
        good = [record(-.4, -.5, m, 'a') for m in INTERNAL]
        good += [record(-.1, .4, m, 'b') for m in INTERNAL]
        self.assertTrue(internal_gate(good, t, .9)['passed'])

    def test_new_prompt_is_predicted_not_oracle(self):
        pool = ('mistral-7b', 'yi-34b')
        scores = torch.tensor([[[.8, .7], [.2, .6], [.1, .5], [.0, .4]]])
        truth = torch.zeros(1, len(PROMPTS), len(pool), 2)
        truth[0, 0, 0, 0] = .5
        truth[0, 0, 1, 0] = .1
        truth[0, 1, 1, 0] = 1.  # The oracle prompt must not be used.
        rows = policy_records([dict(qid='q', task='gsm8k')], scores, truth, pool, 'yi-34b', .9)
        self.assertEqual(rows[0]['new_prompt'], 'direct')
        self.assertAlmostEqual(rows[0]['delta_R'], -.4)


if __name__ == '__main__':
    unittest.main()

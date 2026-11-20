import unittest
import torch
from router.support_graph import SupportGraphRouter
from train_router_support_graph import outcomes, select_support


class SupportGraphTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.net = SupportGraphRouter(3, 2, 4, 6, hidden=8).eval()
        self.s = torch.randn(3, 5)
        self.p = torch.randn(4, 4)
        self.m = torch.randn(3, 6)
        self.o = torch.rand(3, 4, 3, 2)
        self.mask = torch.ones(3, 4, 3, dtype=torch.bool)
        self.q = torch.randn(2, 5)

    def score(self, obs=None, mask=None, models=None):
        return self.net.score(self.q, self.net.encode_support(
            self.s, self.p, self.m if models is None else models,
            self.o if obs is None else obs, self.mask if mask is None else mask))

    def test_masked_outcomes_and_seen_scores_invariant(self):
        mask = self.mask.clone(); mask[:, :, -1] = False
        changed = self.o.clone(); changed[:, :, -1] = float('nan')
        self.assertTrue(torch.equal(self.score(mask=mask), self.score(changed, mask)))
        small = self.score(self.o[:, :, :2], self.mask[:, :, :2], self.m[:2])
        self.assertTrue(torch.allclose(small, self.score()[:, :, :2], atol=1e-6))

    def test_support_changes_new_model_scores(self):
        mask = self.mask.clone(); mask[:, :, -1] = False
        self.assertFalse(torch.allclose(self.score(mask=mask)[:, :, -1], self.score()[:, :, -1]))

    def test_target_isolation_and_gradients(self):
        encoded = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask)
        together = self.net.score(self.q, encoded)
        alone = self.net.score(self.q[1:], encoded)
        self.assertTrue(torch.allclose(together[1:], alone, atol=1e-6))
        together.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.net.parameters()))

    def test_prompt_specific_outcomes_and_pool_permutation(self):
        changed = self.o.clone(); changed[:, 0, 2, 0] += 3
        base = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask)[1]
        after = self.net.encode_support(self.s, self.p, self.m, changed, self.mask)[1]
        self.assertTrue(torch.equal(base[1:], after[1:]))
        self.assertFalse(torch.equal(base[0, 2], after[0, 2]))
        order = torch.tensor([2, 0, 1])
        permuted = self.score(self.o[:, :, order], self.mask[:, :, order], self.m[order])
        self.assertTrue(torch.allclose(permuted, self.score()[:, :, order], atol=1e-6))

    def test_fixed_support_and_seen_only_normalization(self):
        self.assertEqual(select_support(set(map(str, range(30))), 8, 1),
                         select_support(set(map(str, range(30))), 8, 1))
        row = {'action_edges': [dict(prompt=p, model=m, performance=.5,
                                    cost_norm_query=c)
                              for p in ('direct', 'cot', 'decompose', 'selfcheck')
                              for m, c in (('seenA', .2), ('seenB', .6), ('held', 1.))]}
        first = outcomes([row], ('seenA', 'seenB'), True)
        for edge in row['action_edges']:
            edge['cost_norm_query'] = .3 + .4*edge['cost_norm_query']
        second = outcomes([row], ('seenA', 'seenB'), True)
        self.assertTrue(torch.allclose(first, second, atol=1e-6))


class QueryConditionedTest(SupportGraphTest):
    def setUp(self):
        super().setUp()
        self.net.support_weighting = 'query_cosine'

    def score(self, obs=None, mask=None, models=None):
        return self.net.score(self.q, self.net.encode_support(
            self.s, self.p, self.m if models is None else models,
            self.o if obs is None else obs, self.mask if mask is None else mask,
            target_x=self.q))

    def test_target_isolation_and_gradients(self):
        encoded = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask, self.q)
        together = self.net.score(self.q, encoded)
        one = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask, self.q[1:])
        self.assertTrue(torch.allclose(together[1:], self.net.score(self.q[1:], one), atol=1e-6))
        together.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.net.parameters()))

    def test_prompt_specific_outcomes_and_pool_permutation(self):
        changed = self.o.clone(); changed[:, 0, 2, 0] += 3
        base = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask, self.q)[1]
        after = self.net.encode_support(self.s, self.p, self.m, changed, self.mask, self.q)[1]
        self.assertTrue(torch.equal(base[:, 1:], after[:, 1:]))
        self.assertFalse(torch.equal(base[:, 0, 2], after[:, 0, 2]))
        order = torch.tensor([2, 0, 1])
        permuted = self.score(self.o[:, :, order], self.mask[:, :, order], self.m[order])
        self.assertTrue(torch.allclose(permuted, self.score()[:, :, order], atol=1e-6))

    def test_query_changes_support_and_equal_similarity_matches_uniform(self):
        states = self.net.encode_support(self.s, self.p, self.m, self.o, self.mask, self.q)[1]
        self.assertFalse(torch.allclose(states[0], states[1]))
        support = self.s.clone(); support[:, :3] = support[0, :3].clone()
        weighted = self.net.encode_support(support, self.p, self.m, self.o, self.mask, self.q)[1]
        self.net.support_weighting = 'uniform'
        uniform = self.net.encode_support(support, self.p, self.m, self.o, self.mask)[1]
        self.assertTrue(torch.allclose(weighted, uniform[None].expand_as(weighted), atol=1e-6))


if __name__ == '__main__': unittest.main()

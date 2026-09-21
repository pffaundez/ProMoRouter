import ast
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # Allows static CI to report the missing ML extra.
    torch = None

if torch is not None:
    from router.edgegnn_v2_stage_a import (
        FlatSharedObjectiveRouter, MatchedEdgeRouter, build_ego_graph_batch,
        shared_objective, topology_metadata,
    )


class StageAStaticTest(unittest.TestCase):
    def test_graph_builder_has_no_outcome_inputs(self):
        source = Path("router/edgegnn_v2_stage_a.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_ego_graph_batch")
        arguments = {arg.arg for arg in function.args.args}
        self.assertTrue({"query_ids", "query_vectors", "task_vectors", "prompt_vectors", "model_vectors"} <= arguments)
        self.assertFalse({"reward", "performance", "cost", "tokens"} & arguments)

    def test_manifest_freezes_three_by_five_matrix(self):
        import json
        manifest = json.loads(Path("configs/edgegnn_v2_stage_a.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["arms"]), 3)
        self.assertEqual(manifest["data"]["seeds"], [1, 2, 3, 4, 5])
        self.assertFalse(manifest["graph"]["cross_query_message_passing"])
        self.assertFalse(manifest["graph"]["reward_selected_topology"])


@unittest.skipIf(torch is None, "PyTorch is required for the Stage A CPU smoke")
class StageATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.graph = build_ego_graph_batch(
            ["q1", "q2"], torch.randn(2, 5), torch.randn(2, 3),
            torch.randn(4, 7), torch.randn(9, 11),
        )

    def test_exact_disjoint_topology(self):
        self.assertEqual(self.graph.x_dict["query"].shape[0], 2)
        self.assertEqual(self.graph.x_dict["task"].shape[0], 2)
        self.assertEqual(self.graph.x_dict["prompt"].shape[0], 8)
        self.assertEqual(self.graph.x_dict["model"].shape[0], 18)
        expected = {"query_to_task": 2, "task_to_query": 2,
                    "query_to_prompt": 8, "prompt_to_query": 8,
                    "query_to_model": 18, "model_to_query": 18}
        for relation, count in expected.items():
            self.assertEqual(self.graph.edge_index_dict[relation].shape, (2, count))
        # Every copied prompt/model node belongs only to its own query block.
        for relation, width in (("query_to_prompt", 4), ("query_to_model", 9)):
            source, target = self.graph.edge_index_dict[relation]
            self.assertTrue(torch.equal(source, torch.div(target, width, rounding_mode="floor")))
        self.assertEqual(topology_metadata(2, 4, 9)["directed_edges_per_query"], 28)

    def test_all_36_scores_and_no_mp_bypass(self):
        dims = {"query": 5, "task": 3, "prompt": 7, "model": 11}
        full = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=True)
        self_only = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=False)
        self_only.load_state_dict(full.state_dict())
        full.eval(); self_only.eval()
        self.assertEqual(full(self.graph).shape, (2, 36))
        before = self_only(self.graph)
        changed_edges = {name: value.clone() for name, value in self.graph.edge_index_dict.items()}
        changed_edges["task_to_query"] = torch.tensor([[1, 0], [0, 1]])
        changed = type(self.graph)(self.graph.x_dict, changed_edges, self.graph.query_ids,
                                   self.graph.prompt_count, self.graph.model_count)
        self.assertTrue(torch.equal(before, self_only(changed)))
        self.assertFalse(torch.equal(full(self.graph), full(changed)))

    def test_ego_graph_isolation(self):
        dims = {"query": 5, "task": 3, "prompt": 7, "model": 11}
        model = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=True).eval()
        before = model(self.graph)
        changed = build_ego_graph_batch(
            ["q1", "q2"],
            torch.stack((self.graph.x_dict["query"][0] * 100, self.graph.x_dict["query"][1])),
            self.graph.x_dict["task"], self.graph.x_dict["prompt"][:4], self.graph.x_dict["model"][:9],
        )
        after = model(changed)
        self.assertFalse(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_expected_gradient_activity(self):
        dims = {"query": 5, "task": 3, "prompt": 7, "model": 11}
        rewards = torch.randn(2, 36)
        full = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=True)
        self_only = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=False)
        shared_objective(full(self.graph), rewards).backward()
        shared_objective(self_only(self.graph), rewards).backward()
        self.assertTrue(any(p.grad is not None for name, p in full.named_parameters() if "rel_linear" in name))
        self.assertTrue(all(p.grad is None for name, p in self_only.named_parameters() if "rel_linear" in name))
        self.assertTrue(all(p.grad is not None for name, p in self_only.named_parameters() if "self_linear" in name))

    def test_pre_message_pair_representation_is_identical(self):
        dims = {"query": 5, "task": 3, "prompt": 7, "model": 11}
        full = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=True)
        self_only = MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=False)
        self_only.load_state_dict(full.state_dict())
        full_initial = {n: full.projections[n](self.graph.x_dict[n]) for n in ("query", "task", "prompt", "model")}
        self_initial = {n: self_only.projections[n](self.graph.x_dict[n]) for n in ("query", "task", "prompt", "model")}
        self.assertTrue(torch.equal(full.compose_edges(full_initial, self.graph),
                                    self_only.compose_edges(self_initial, self.graph)))

    def test_cpu_smoke_all_arms(self):
        dims = {"query": 5, "task": 3, "prompt": 7, "model": 11}
        arms = [
            MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=True),
            MatchedEdgeRouter(dims, hidden_dim=16, dropout=0.0, message_passing=False),
            FlatSharedObjectiveRouter(5, 3, 7, 11, hidden_dim=16, dropout=0.0),
        ]
        rewards = torch.randn(2, 36)
        for model in arms:
            loss = shared_objective(model(self.graph), rewards)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_labels_cannot_enter_graph_builder(self):
        names = build_ego_graph_batch.__code__.co_varnames[:build_ego_graph_batch.__code__.co_argcount]
        for forbidden in ("reward", "performance", "cost", "tokens"):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()

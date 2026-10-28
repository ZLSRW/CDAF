"""Behavior tests for original records, isolation, missing evidence and DDIM gradients."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from cdaf.config import build_parser
from cdaf.data import DiseasePaths, load_dataset, make_split, mask_target_edges, pair_codes
from cdaf.diffusion import LatentDiffusion
from cdaf.metrics import aggregate_pairs, choose_threshold, evaluate_metrics
from cdaf.model import CDAF, classification_loss


ROOT = Path(__file__).resolve().parents[1]


def small_args():
    return build_parser().parse_args(
        [
            "--hidden-dim",
            "16",
            "--diffusion-width",
            "32",
            "--diffusion-blocks",
            "2",
            "--bilinear-rank",
            "8",
            "--max-paths",
            "4",
            "--ddim-start",
            "10",
            "--ddim-steps",
            "2",
            "--dropout",
            "0",
        ]
    )


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = {
            name: load_dataset(ROOT / "datasets", name)
            for name in ("B-dataset", "C-dataset", "F-dataset")
        }

    def test_original_records_and_grouped_splits(self):
        for name, data in self.datasets.items():
            with self.subTest(dataset=name):
                original = pd.read_csv(
                    ROOT / "datasets" / name / "DrugProteinAssociationNumber.csv"
                )
                np.testing.assert_array_equal(data.dt, original[["drug", "protein"]].to_numpy())
                split = make_split(data, small_args(), 0)
                retained = np.concatenate([v[v >= 0] for v in split.source_rows.values()])
                np.testing.assert_array_equal(np.sort(retained), np.arange(len(data.dt)))
                sets = {k: set(pair_codes(v, data.n_proteins)) for k, v in split.pairs.items()}
                self.assertFalse(sets["train"] & sets["val"])
                self.assertFalse(sets["train"] & sets["test"])
                self.assertFalse(sets["val"] & sets["test"])
                positives = set(pair_codes(data.dt, data.n_proteins))
                for key in split.pairs:
                    negative = split.pairs[key][split.labels[key] == 0]
                    self.assertFalse(set(pair_codes(negative, data.n_proteins)) & positives)

    def test_cold_splits_and_strict_context(self):
        data = self.datasets["B-dataset"]
        for setting in ("cold-drug", "cold-protein", "cold-both"):
            with self.subTest(split=setting):
                args = small_args()
                args.split, args.cold_context = setting, "strict"
                split = make_split(data, args, 0)
                for col, affected in [(0, setting != "cold-protein"), (1, setting != "cold-drug")]:
                    if affected:
                        tr = set(split.pairs["train"][:, col])
                        va = set(split.pairs["val"][:, col])
                        te = set(split.pairs["test"][:, col])
                        self.assertFalse(tr & va or tr & te or va & te)
                        aux = split.ds if col == 0 else split.ps
                        self.assertFalse(set(aux[:, 0]) & (va | te))

    def test_outer_test_folds_cover_each_positive_group_once(self):
        data, args = self.datasets["F-dataset"], small_args()
        seen = set()
        seen_negatives = set()
        for fold in range(args.folds):
            split = make_split(data, args, fold)
            test = split.pairs["test"][split.labels["test"] == 1]
            current = set(pair_codes(test, data.n_proteins))
            self.assertFalse(current & seen)
            seen.update(current)
            negatives = split.pairs["test"][split.labels["test"] == 0]
            negative_codes = set(pair_codes(negatives, data.n_proteins))
            self.assertFalse(negative_codes & seen_negatives)
            seen_negatives.update(negative_codes)
        self.assertEqual(seen, set(pair_codes(data.dt, data.n_proteins)))

    def test_f_node_identifiers_follow_feature_order(self):
        data = self.datasets["F-dataset"]
        nodes = pd.read_csv(ROOT / "datasets/F-dataset/AllNode.csv")["id"]
        self.assertEqual(data.drug_ids, nodes.iloc[: data.n_drugs].tolist())

    def test_mask_removes_all_copies_of_supervised_positive(self):
        edges = torch.tensor([[0, 1], [0, 1], [1, 0], [2, 1]])
        retained = mask_target_edges(
            edges, torch.tensor([[0, 1], [2, 0]]), torch.tensor([1.0, 0.0]), 2
        )
        self.assertEqual(retained.tolist(), [[1, 0], [2, 1]])
        self.assertEqual(len(edges), 4)

    def test_path_multiplicity_and_missing_paths(self):
        ds = np.array([[0, 0], [0, 0], [0, 1]])
        ps = np.array([[0, 0], [0, 0], [0, 0], [0, 1]])
        sampler = DiseasePaths(ds, ps, 2, 2, 2, 10, 42)
        ids, valid, counts = sampler.batch(np.array([[0, 0], [1, 1]]), "cpu")
        self.assertEqual(counts.tolist(), [7, 0])
        self.assertEqual(int((ids[0][valid[0]] == 0).sum()), 6)
        self.assertEqual(int(valid[1].sum()), 0)
        again = sampler.batch(np.array([[1, 1], [0, 0]]), "cpu")
        torch.testing.assert_close(ids[0], again[0][1])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(5)
        self.args = small_args()
        self.tensors = dict(
            drug=torch.randn(3, 300),
            protein=torch.randn(3, 320),
            disease=torch.randn(2, 64),
            ds=torch.tensor([[0, 0], [1, 1]]),
            ps=torch.tensor([[0, 0], [1, 1]]),
        )
        self.paths = DiseasePaths(
            self.tensors["ds"].numpy(), self.tensors["ps"].numpy(), 3, 3, 2, 4, 42
        )
        self.degree = torch.tensor(self.paths.degree, dtype=torch.float32)

    def test_missing_evidence_is_masked_and_training_is_finite(self):
        model = CDAF(self.args)
        pairs = torch.tensor([[0, 0], [2, 2]])
        paths = self.paths.batch(pairs.numpy(), "cpu")
        output = model(
            self.tensors,
            pairs,
            torch.empty((0, 2), dtype=torch.long),
            paths,
            self.degree,
            use_diffusion=False,
        )
        torch.testing.assert_close(output["gate"][1], torch.tensor([1.0, 0.0, 0.0]))
        self.assertTrue(torch.isfinite(output["probability"]).all())
        loss = classification_loss(output, torch.tensor([1.0, 0.0]), 0.1)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(model.path.transformer.linear1.weight.grad.abs().sum()), 0)

    def test_ddim_preserves_gradients_and_matches_oracle(self):
        diffusion = LatentDiffusion(self.args)
        x = torch.randn(8, self.args.hidden_dim)
        diffusion.fit_stats(x, x, torch.arange(8), torch.arange(8))
        c, available = torch.randn_like(x), torch.ones(len(x))
        restored = diffusion.enhance(x, c, available, 0)
        restored.square().mean().backward()
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in diffusion.denoiser.parameters()
                if p.grad is not None
            ),
            0,
        )
        epsilon = torch.randn_like(x)

        class Oracle(torch.nn.Module):
            def forward(self, z, t, kind, context, available):
                return epsilon

        diffusion.denoiser = Oracle()
        restored = diffusion.enhance(x, c, available, 0, noise=epsilon)
        torch.testing.assert_close(restored, x, atol=2e-6, rtol=2e-6)
        self.assertEqual(float(diffusion.alpha_bar[0]), 1.0)
        self.assertTrue(torch.all(diffusion.alpha_bar[:-1] > diffusion.alpha_bar[1:]))

    def test_checkpoint_round_trip(self):
        model = CDAF(self.args).eval()
        pairs = torch.tensor([[0, 0], [1, 1], [2, 2]])
        paths = self.paths.batch(pairs.numpy(), "cpu")
        encoded = model.encode(self.tensors)
        model.diffusion.fit_stats(
            encoded["drug"], encoded["protein"], torch.arange(3), torch.arange(3)
        )
        edges = torch.tensor([[0, 1], [1, 0]])
        first = model(self.tensors, pairs, edges, paths, self.degree, noise_seed=123)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "state.pt"
            torch.save(model.state_dict(), file)
            loaded = CDAF(self.args).eval()
            loaded.load_state_dict(torch.load(file, weights_only=True))
            second = loaded(self.tensors, pairs, edges, paths, self.degree, noise_seed=123)
        torch.testing.assert_close(first["probability"], second["probability"])
        torch.testing.assert_close(first["gate"].sum(1), torch.ones(3))

    def test_freezing_disables_dropout_and_encoder_gradients(self):
        model = CDAF(self.args)
        for parameter in model.encoders.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.freeze_encoders()
        model.train()
        self.assertFalse(model.encoders.training)
        self.assertTrue(model.path.training)
        for module in (model.encoders.drug, model.encoders.protein, model.encoders.disease):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.condition_parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoders.parameters()))

    def test_disabled_channels_reduce_to_bilinear(self):
        args = self.args
        args.disable_graph = args.disable_path = True
        model = CDAF(args).eval()
        pairs = torch.tensor([[0, 0]])
        output = model(
            self.tensors,
            pairs,
            torch.tensor([[0, 0]]),
            self.paths.batch(pairs.numpy(), "cpu"),
            self.degree,
            use_diffusion=False,
        )
        torch.testing.assert_close(output["gate"], torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(output["probability"], output["channel_logits"][:, 0].sigmoid())

    def test_single_predictor_removes_channels_and_keeps_diffusion_gradients(self):
        self.args.single_predictor = True
        model = CDAF(self.args)
        for name in ("bilinear", "graph", "path", "router", "router_projections"):
            self.assertFalse(hasattr(model, name))
        model.freeze_encoders()
        encoded = model.encode(self.tensors)
        model.diffusion.fit_stats(
            encoded["drug"], encoded["protein"], torch.arange(3), torch.arange(3)
        )
        pairs = torch.tensor([[0, 0], [1, 2], [2, 1]])
        output = model(self.tensors, pairs, None, None, None, encoded=encoded, noise_seed=123)
        self.assertEqual(set(output), {"probability", "single_logit"})
        labels = torch.tensor([1.0, 0.0, 1.0])
        loss = classification_loss(output, labels, 0.1)
        torch.testing.assert_close(loss, classification_loss(output, labels, 100.0))
        loss.backward()
        for module in (model.encoders.drug, model.encoders.protein, model.encoders.disease):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        for module in (model.diffusion.denoiser, model.predictor):
            self.assertGreater(
                sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0
            )
        loaded = CDAF(self.args).eval()
        loaded.load_state_dict(model.state_dict())
        model.eval()
        first = model(self.tensors, pairs, None, None, None, noise_seed=123)
        second = loaded(self.tensors, pairs, None, None, None, noise_seed=123)
        torch.testing.assert_close(first["probability"], second["probability"])

    def test_no_diffusion_is_identity_and_has_no_diffusion_gradients(self):
        self.args.disable_diffusion = True
        model = CDAF(self.args)
        encoded = model.encode(self.tensors)
        enhanced = model.enhance(encoded, True)
        self.assertIs(enhanced[0], encoded["drug"])
        self.assertIs(enhanced[1], encoded["protein"])
        self.assertTrue(all(not p.requires_grad for p in model.diffusion.parameters()))
        pairs = torch.tensor([[0, 0], [1, 2]])
        output = model(
            self.tensors,
            pairs,
            torch.tensor([[0, 1]]),
            self.paths.batch(pairs.numpy(), "cpu"),
            self.degree,
        )
        classification_loss(output, torch.tensor([1.0, 0.0]), 0.1).backward()
        self.assertTrue(all(p.grad is None for p in model.diffusion.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.bilinear.parameters()))


class MetricsTests(unittest.TestCase):
    def test_record_and_pair_metrics_are_explicit(self):
        pairs = np.array([[0, 0], [0, 0], [0, 1], [1, 1]])
        labels = np.array([1, 1, 0, 1])
        scores = np.array([0.8, 0.8, 0.2, 0.7])
        grouped = aggregate_pairs(pairs, labels, scores)
        threshold = choose_threshold(grouped.label, grouped.probability)
        metrics = evaluate_metrics(pairs, labels, scores, threshold)
        self.assertEqual(metrics["pairs"]["n"], 3)
        self.assertEqual(metrics["records"]["n"], 4)
        self.assertEqual(set(metrics), {"pairs", "records"})
        self.assertEqual(
            set(metrics["pairs"]), {"auc", "aupr", "accuracy", "f1", "threshold", "n", "positives"}
        )
        for key in ["auc", "aupr", "accuracy", "f1"]:
            self.assertEqual(metrics["pairs"][key], 1.0)


if __name__ == "__main__":
    unittest.main()

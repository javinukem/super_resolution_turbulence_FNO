"""Smoke tests for the unified experiment engine (src/training/experiment.py).

GPU-free, data-free: validates that the engine module imports cleanly and
that every experiments/*/experiment.yaml parses and satisfies the engine's
config schema.
"""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TOP = {"experiment", "output", "data", "training", "runs"}
REQUIRED_RUN = {"name", "label", "model_type", "loss"}
MODEL_TYPES = {"sfno", "edsr", "usfno"}
LOSS_TYPES = {
    "mse", "l1", "spectral", "mse_spectral", "mse_l1", "mse_spectral_l1",
    "l1_spectral",
}
REQUIRED_TRAINING = {
    "epochs", "learning_rate", "weight_decay", "batch_size", "grad_accum",
    "noise_std", "early_stop_patience", "sched_factor", "sched_patience",
}


class ExperimentEngineTest(unittest.TestCase):
    """Import the engine without touching CUDA (autocvd is CLI-gated)."""

    def test_engine_imports_gpu_free(self):
        import sys

        sys.path.insert(0, str(ROOT))
        import src.training.experiment as engine  # noqa: F401

        self.assertTrue(callable(engine.build_loss_fn))
        self.assertTrue(callable(engine.train_one_run))


class ExperimentConfigTest(unittest.TestCase):
    """Validate every experiment.yaml against the engine's schema."""

    def _configs(self):
        return sorted(ROOT.glob("experiments/*/experiment.yaml"))

    def test_configs_exist(self):
        # The 4 published-model-backed experiments (experiments/MODELS.md)
        self.assertGreaterEqual(len(self._configs()), 4)

    def test_configs_are_valid(self):
        for path in self._configs():
            with self.subTest(config=path):
                with open(path) as f:
                    cfg = yaml.safe_load(f)
                self.assertTrue(REQUIRED_TOP <= set(cfg), path)
                self.assertIn(cfg["experiment"], str(path.parent.name))
                for key in ("scratch_root", "folder_prefix", "repo_dir"):
                    self.assertIn(key, cfg["output"], path)
                for key in ("train_h5", "val_h5", "snapshot_index", "upsample_factor"):
                    self.assertIn(key, cfg["data"], path)
                self.assertTrue(REQUIRED_TRAINING <= set(cfg["training"]), path)
                self.assertGreaterEqual(len(cfg["runs"]), 1, path)
                for run in cfg["runs"]:
                    self.assertTrue(REQUIRED_RUN <= set(run), (path, run.get("name")))
                    self.assertIn(run["model_type"], MODEL_TYPES, path)
                    self.assertIn(run["loss"]["type"], LOSS_TYPES, path)
                    self.assertIsInstance(run["model_params"], dict, path)
                    self.assertNotIn(
                        "apply_positivity_relu", run["model_params"], path
                    )
                for ref in cfg.get("references") or []:
                    self.assertIn("adopt_from", ref, (path, ref))
                    self.assertIn("scratch_glob", ref["adopt_from"], (path, ref))
                for key in ("spectral_pool", "trilinear_baseline"):
                    self.assertIn(key, cfg, path)

    def test_run_names_are_unique(self):
        for path in self._configs():
            with open(path) as f:
                cfg = yaml.safe_load(f)
            names = [r["name"] for r in cfg["runs"]]
            self.assertEqual(len(names), len(set(names)), path)


if __name__ == "__main__":
    unittest.main()

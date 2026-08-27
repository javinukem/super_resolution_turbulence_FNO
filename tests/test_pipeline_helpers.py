import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

from src.utils.pipeline import (
    build_dataset,
    build_model,
    build_model_tag,
    create_run_dir,
    instantiate_model,
    save_training_artifacts,
)


class TestPipelineHelpers(unittest.TestCase):
    def test_build_model_tag_formats_bool(self):
        cfg = OmegaConf.create(
            {
                "name": "fno_2",
                "params": {"modes": 16, "apply_constraint": False},
                "tag_template": "m{modes}_ac{apply_constraint_int}",
            }
        )
        self.assertEqual(build_model_tag(cfg), "m16_ac0")

    def test_create_run_dir_creates_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            runtime_cfg = OmegaConf.create(
                {
                    "timestamp_format": "%Y",
                    "date_format": "%m",
                    "output": {
                        "root_dir": tmp_dir,
                        "run_dir_template": "runs/{date}_{model_tag}",
                    },
                }
            )
            run_dir, context = create_run_dir(runtime_cfg, "demo")
            self.assertTrue(run_dir.exists())
            self.assertEqual(run_dir.parent.name, "runs")
            self.assertTrue(run_dir.name.endswith("_demo"))
            self.assertIn("date", context)
            self.assertIn("timestamp", context)

    def test_save_training_artifacts_writes_config_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            runtime_cfg = OmegaConf.create(
                {
                    "output": {
                        "config_name": "config.yaml",
                        "weights_name_template": "weights_{date}.pt",
                        "save_config": True,
                        "save_weights": True,
                        "save_loss_curve": False,
                    }
                }
            )
            model = torch.nn.Linear(1, 1)
            save_training_artifacts(
                run_dir=run_dir,
                model=model,
                train_losses=[1.0, 0.5],
                test_losses=[1.1, 0.7],
                config_payload={"hello": "world"},
                runtime_cfg=runtime_cfg,
                context={"date": "2026-01-01", "timestamp": "ts", "model_tag": "x"},
            )
            config_path = run_dir / "config.yaml"
            weights_path = run_dir / "weights_2026-01-01.pt"
            self.assertTrue(config_path.exists())
            self.assertTrue(weights_path.exists())
            with open(config_path, "r", encoding="utf-8") as f:
                payload = yaml.safe_load(f)
            self.assertEqual(payload["hello"], "world")

    def test_build_dataset_invalid_kind_raises(self):
        cfg = OmegaConf.create({"kind": "unsupported", "kwargs": {}})
        with self.assertRaises(ValueError):
            build_dataset(cfg)

    def test_instantiate_model_and_build_model_from_dynamic_module(self):
        module_name = "tests._dummy_module"
        dummy_module = types.ModuleType(module_name)

        class DummyModel(torch.nn.Module):
            def __init__(self, size=3):
                super().__init__()
                self.layer = torch.nn.Linear(size, 1)

        dummy_module.DummyModel = DummyModel
        sys.modules[module_name] = dummy_module
        try:
            model = instantiate_model(module_name, "DummyModel", {"size": 2})
            self.assertIsInstance(model, DummyModel)

            cfg = OmegaConf.create(
                {
                    "module": module_name,
                    "class_name": "DummyModel",
                    "params": {"size": 4},
                }
            )
            built_model = build_model(cfg)
            self.assertIsInstance(built_model, DummyModel)
            self.assertEqual(built_model.layer.in_features, 4)
        finally:
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()

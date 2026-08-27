import subprocess
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestConfigLayout(unittest.TestCase):
    def test_required_config_groups_exist(self):
        required_dirs = [
            REPO_ROOT / "configs" / "model",
            REPO_ROOT / "configs" / "data",
            REPO_ROOT / "configs" / "training",
            REPO_ROOT / "configs" / "runtime",
            REPO_ROOT / "configs" / "preset",
        ]
        for path in required_dirs:
            self.assertTrue(path.exists(), f"Missing config group directory: {path}")

    def test_base_config_contains_hydra_and_defaults(self):
        config_path = REPO_ROOT / "configs" / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("defaults", cfg)
        self.assertIn("hydra", cfg)
        defaults_repr = str(cfg["defaults"])
        for expected in ["model", "data", "training", "runtime", "preset"]:
            self.assertIn(expected, defaults_repr)


class TestHydraCliSmoke(unittest.TestCase):
    def _run_cmd(self, *args):
        cmd = [sys.executable, *args, "--cfg", "job"]
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_train_presets_cfg_compose(self):
        self._run_cmd("train.py")
        self._run_cmd("train.py", "preset=dsfno")
        self._run_cmd("train.py", "preset=cnn")
        self._run_cmd("train.py", "preset=fno_2_2d")
        self._run_cmd("train.py", "preset=fno_2_exp4")

    def test_refactored_entrypoints_cfg_compose(self):
        self._run_cmd("evaluation/experiment_2/cfno_2_eval.py")
        self._run_cmd("experiments/grid_search_dsfno.py")
        self._run_cmd("experiments/fno_2d_grid_search/grid_search_fno2_2d.py")
        self._run_cmd("experiments/cfno_2/experiment_6/training_fno.py")


if __name__ == "__main__":
    unittest.main()

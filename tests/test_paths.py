"""Tests for the cluster-agnostic path resolution (src/utils/paths.py)."""

import importlib
import os
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class PathsTest(unittest.TestCase):
    ENV_VARS = (
        "TURBULENCE_SR_DATA",
        "TURBULENCE_SR_SOURCE_DATA",
        "TURBULENCE_SR_SCRATCH",
    )

    def _reload(self):
        import src.utils.paths as paths

        return importlib.reload(paths)

    def test_defaults_are_repo_relative(self):
        with mock.patch.dict(os.environ, clear=False):
            for var in self.ENV_VARS:
                os.environ.pop(var, None)
            paths = self._reload()
        self.assertEqual(paths.DATA_DIR, ROOT / "data/full_states_h5")
        self.assertEqual(paths.TRAIN_H5, ROOT / "data/full_states_h5/full_states.h5")
        self.assertEqual(paths.VAL_H5, ROOT / "data/full_states_h5/full_states_val.h5")
        self.assertEqual(paths.SOURCE_DATA_DIR, ROOT / "data/full_states_h5")
        self.assertEqual(paths.SCRATCH_ROOT, ROOT / "runs/experiments")

    def test_env_overrides(self):
        with mock.patch.dict(
            os.environ,
            {
                "TURBULENCE_SR_DATA": "/somewhere/data",
                "TURBULENCE_SR_SOURCE_DATA": "/somewhere/source",
                "TURBULENCE_SR_SCRATCH": "/somewhere/scratch",
            },
        ):
            paths = self._reload()
        self.assertEqual(paths.DATA_DIR, Path("/somewhere/data"))
        self.assertEqual(paths.TRAIN_H5, Path("/somewhere/data/full_states.h5"))
        self.assertEqual(paths.VAL_H5, Path("/somewhere/data/full_states_val.h5"))
        self.assertEqual(paths.SOURCE_DATA_DIR, Path("/somewhere/source"))
        self.assertEqual(paths.SCRATCH_ROOT, Path("/somewhere/scratch"))


if __name__ == "__main__":
    unittest.main()

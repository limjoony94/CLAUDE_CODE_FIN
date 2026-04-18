#!/usr/bin/env python3
"""Unit tests for pipeline.py core functions."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "automation"))
sys.path.insert(0, str(ROOT / "scripts" / "production"))

import pipeline


class TestSafeIO(unittest.TestCase):
    """Test OneDrive-safe JSON read/write."""

    def test_safe_read_existing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            result = pipeline._safe_read_json(path)
            self.assertEqual(result["key"], "value")
        finally:
            os.unlink(path)

    def test_safe_read_missing(self):
        result = pipeline._safe_read_json("/nonexistent.json", default={"default": True})
        self.assertEqual(result["default"], True)

    def test_safe_read_corrupt(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json{{{")
            path = f.name
        try:
            result = pipeline._safe_read_json(path, default={"fallback": 1}, retries=1)
            self.assertEqual(result["fallback"], 1)
        finally:
            os.unlink(path)

    def test_safe_write_and_read(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            pipeline._safe_write_json(path, {"test": 42})
            result = pipeline._safe_read_json(path)
            self.assertEqual(result["test"], 42)
        finally:
            os.unlink(path)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity checker."""

    def test_clean_data(self):
        import pandas as pd
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close\n")
            f.write("2026-01-01 00:00:00,100,101,99,100.5\n")
            f.write("2026-01-01 00:05:00,100.5,102,100,101\n")
            path = f.name
        try:
            issues = pipeline._check_data_integrity(path)
            self.assertEqual(len(issues), 0)
        finally:
            os.unlink(path)

    def test_duplicate_data(self):
        import pandas as pd
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("timestamp,open,high,low,close\n")
            f.write("2026-01-01 00:00:00,100,101,99,100.5\n")
            f.write("2026-01-01 00:00:00,100,101,99,100.5\n")
            path = f.name
        try:
            issues = pipeline._check_data_integrity(path)
            self.assertTrue(any("duplicate" in i for i in issues))
        finally:
            os.unlink(path)


class TestManifest(unittest.TestCase):
    """Test deploy manifest."""

    def test_manifest_exists(self):
        self.assertTrue(pipeline.DEPLOY_MANIFEST.exists(),
                       "deploy_manifest.json should exist")

    def test_manifest_structure(self):
        manifest = pipeline._safe_read_json(pipeline.DEPLOY_MANIFEST)
        self.assertIn("version", manifest)
        self.assertIn("deployed_at", manifest)
        self.assertIn("config_snapshot", manifest)
        self.assertIn("history", manifest)

    def test_config_version_from_manifest(self):
        version = pipeline._get_config_version()
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "unknown")

    def test_config_epoch_from_manifest(self):
        epoch = pipeline._get_config_epoch()
        self.assertIsInstance(epoch, str)
        # Should be parseable as datetime
        datetime.fromisoformat(epoch)


class TestProductionParams(unittest.TestCase):
    """Test production config loading."""

    def test_params_loaded(self):
        params = pipeline._get_production_params()
        self.assertIn("n_slots", params)
        self.assertIn("direction_cap", params)
        self.assertIn("timeout_bars", params)
        self.assertEqual(params["n_slots"], 7)
        self.assertEqual(params["direction_cap"], 7)

    def test_cascade_off(self):
        params = pipeline._get_production_params()
        self.assertEqual(params["cascade_tighten_pct"], 0)
        self.assertEqual(params["preemptive_cascade_pct"], 999)


class TestValidation(unittest.TestCase):
    """Test scan validation logic."""

    def test_reject_few_patterns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"pattern_details": {"a": {}, "b": {}}}, f)
            path = f.name
        try:
            self.assertFalse(pipeline._validate_scan(path))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

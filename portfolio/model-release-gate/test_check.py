"""Run from this directory with python -m unittest."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReleaseGateTest(unittest.TestCase):
    def run_gate(self, candidate, *, baseline=None, minimum="0.80"):
        if baseline is None:
            baseline = '{"dataset_id":"v2","metrics":{"au_prc":0.84}}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            baseline_path.write_text(baseline, encoding="utf-8")
            candidate_path.write_text(candidate, encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable, str(Path(__file__).with_name("check.py")),
                    str(baseline_path), str(candidate_path), "--metric", "au_prc",
                    "--minimum", minimum, "--max-regression", "0.02",
                ],
                capture_output=True, text=True, timeout=10,
            )

    def test_exact_regression_limit_passes(self):
        result = self.run_gate('{"dataset_id":"v2","metrics":{"au_prc":0.82}}')
        self.assertEqual(result.returncode, 0, result.stderr)
        decision = json.loads(result.stdout)
        self.assertIs(decision["passed"], True)
        self.assertEqual(decision["regression"], "0.02")

    def test_policy_failures_block_promotion(self):
        for value, minimum in [("0.819", "0.80"), ("0.83", "0.85")]:
            with self.subTest(value=value, minimum=minimum):
                report = '{"dataset_id":"v2","metrics":{"au_prc":' + value + '}}'
                result = self.run_gate(report, minimum=minimum)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIs(json.loads(result.stdout)["passed"], False)

    def test_invalid_reports_fail_without_a_decision(self):
        for report, error in [
            ('{"dataset_id":"v3","metrics":{"au_prc":0.90}}', "dataset IDs must match"),
            ('{"dataset_id":"v2","metrics":{}}', "missing metric"),
            ('{"dataset_id":"v2","metrics":{"au_prc":NaN}}', "finite number"),
            ('{"dataset_id":"v2","metrics":{"au_prc":true}}', "finite number"),
            ("{", "Expecting property name"),
        ]:
            with self.subTest(report=report):
                result = self.run_gate(report)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(error, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

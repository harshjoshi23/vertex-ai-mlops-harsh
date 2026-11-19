"""Run from this directory with python -m unittest."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReleaseGateTest(unittest.TestCase):
    def run_gate(self, candidate, *, baseline=None, minimum="0.80", metric="au_prc"):
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
                    str(baseline_path), str(candidate_path), "--metric", metric,
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

    def test_vertex_evaluations(self):
        schema = (
            "gs://google-cloud-aiplatform/schema/modelevaluation/"
            "classification_metrics_1.0.0.yaml"
        )

        def report(score, *, schema_uri=schema, dataset="v2"):
            return json.dumps({
                "dataset_id": dataset,
                "vertex_evaluation": {
                    "metricsSchemaUri": schema_uri,
                    "metrics": {"auPrc": score, "auRoc": score},
                },
            })

        for candidate, metric, status in [
            (report(0.82), "au_prc", 0),
            (report(0.819), "au_prc", 1),
            (report(0.82), "au_roc", 0),
            (report(True), "au_prc", 2),
            (report(0.82, schema_uri="unsupported"), "au_prc", 2),
            (report(0.82, dataset="v3"), "au_prc", 2),
            (report(0.82), "log_loss", 2),
        ]:
            with self.subTest(candidate=candidate, metric=metric):
                result = self.run_gate(candidate, baseline=report(0.84), metric=metric)
                self.assertEqual(result.returncode, status, result.stderr)
                if status == 2:
                    self.assertEqual(result.stdout, "")
                else:
                    self.assertIs(json.loads(result.stdout)["passed"], status == 0)

    def test_invalid_threshold_is_an_input_error(self):
        result = self.run_gate(
            '{"dataset_id":"v2","metrics":{"au_prc":0.9}}', minimum="invalid"
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()

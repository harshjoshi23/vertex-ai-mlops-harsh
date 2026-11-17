"""Exercise report preparation through the promotion check."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
EVALUATION = "projects/123/locations/us-central1/models/456/evaluations/789"


class PrepareReportTest(unittest.TestCase):
    def test_saved_evaluation_to_release_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = root / "response.json"
            baseline = root / "baseline.json"
            baseline.write_text('{"dataset_id":"holdout-v2","metrics":{"au_prc":0.84}}')
            evaluation = {
                "name": EVALUATION,
                "metricsSchemaUri": (
                    "gs://google-cloud-aiplatform/schema/modelevaluation/"
                    "classification_metrics_1.0.0.yaml"
                ),
                "metrics": {"auPrc": 0.82},
            }
            other = dict(evaluation, name=EVALUATION + "0", metrics={"auPrc": 0.1})
            for entries, status in [([other, evaluation], 0), ([], 2), ([evaluation] * 2, 2)]:
                with self.subTest(entries=len(entries), expected_status=status):
                    response.write_text(json.dumps({"modelEvaluations": entries}))
                    result = subprocess.run(
                        [sys.executable, str(HERE / "prepare_report.py"), str(response),
                         "--evaluation", EVALUATION, "--dataset-id", "holdout-v2"],
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, status, result.stderr)
                    if status == 2:
                        self.assertIn("exactly one matching evaluation", result.stderr)
                        self.assertEqual(result.stdout, "")
                        continue

                    candidate = root / "candidate.json"
                    candidate.write_text(result.stdout)
                    self.assertEqual(json.loads(result.stdout)["vertex_evaluation"]["name"], EVALUATION)
                    gate = subprocess.run(
                        [sys.executable, str(HERE / "check.py"), str(baseline), str(candidate),
                         "--metric", "au_prc", "--minimum", "0.80", "--max-regression", "0.02"],
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(gate.returncode, 0, gate.stderr)
                    self.assertIs(json.loads(gate.stdout)["passed"], True)

    def test_duplicate_fields_rejected_by_both_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            cases = [
                (
                    "check.py",
                    '{"dataset_id":"v2","metrics":{"au_prc":0.40,"au_prc":0.95}}',
                    [str(report), "--metric", "au_prc", "--minimum", "0.80",
                     "--max-regression", "0.02"],
                    "au_prc",
                ),
                (
                    "prepare_report.py",
                    '{"modelEvaluations":[{"name":"other","name":"' + EVALUATION + '"}]}',
                    ["--evaluation", EVALUATION, "--dataset-id", "v2"],
                    "name",
                ),
            ]
            for script, payload, arguments, key in cases:
                with self.subTest(script=script):
                    report.write_text(payload, encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, str(HERE / script), str(report), *arguments],
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(f"duplicate JSON key {key!r}", result.stderr)
                    self.assertEqual(result.stdout, "")

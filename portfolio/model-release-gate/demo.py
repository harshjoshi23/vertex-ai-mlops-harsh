"""Run an offline promotion demo with synthetic Vertex AI evaluations.

Run from any directory with python3 /path/to/demo.py.
All reports are temporary. The demo exits successfully after verifying both
a passing candidate and a candidate that must be blocked.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        reports = {}
        for model, score in [("baseline", 0.84), ("passing", 0.83), ("regressing", 0.79)]:
            name = f"projects/demo/locations/us-central1/models/{model}/evaluations/1"
            response = root / f"{model}-evaluations.json"
            response.write_text(json.dumps({"modelEvaluations": [{
                "name": name,
                "metricsSchemaUri": (
                    "gs://google-cloud-aiplatform/schema/modelevaluation/"
                    "classification_metrics_1.0.0.yaml"
                ),
                "metrics": {"auPrc": score},
            }]}), encoding="utf-8")
            prepared = subprocess.run(
                [sys.executable, str(HERE / "prepare_report.py"), str(response),
                 "--evaluation", name, "--dataset-id", "synthetic-holdout-v1"],
                capture_output=True, text=True, check=True, timeout=10,
            )
            reports[model] = root / f"{model}-report.json"
            reports[model].write_text(prepared.stdout, encoding="utf-8")

        for model, expected in [("passing", 0), ("regressing", 1)]:
            result = subprocess.run(
                [sys.executable, str(HERE / "check.py"), str(reports["baseline"]),
                 str(reports[model]), "--metric", "au_prc", "--minimum", "0.80",
                 "--max-regression", "0.02"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != expected:
                raise RuntimeError(f"{model}: unexpected exit {result.returncode}: {result.stderr}")
            decision = json.loads(result.stdout)
            if decision["passed"] != (expected == 0):
                raise RuntimeError(f"{model}: inconsistent release decision")
            print(json.dumps({"candidate": model, "decision": decision}))


if __name__ == "__main__":
    main()

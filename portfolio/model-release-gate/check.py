"""Compare evaluation reports before promoting a model (Python 3.11+).

Run: python check.py baseline.json candidate.json --metric au_prc \
    --minimum 0.80 --max-regression 0.02

Each report is an object such as
{"dataset_id": "holdout-v2", "metrics": {"au_prc": 0.84}}.
For saved Vertex AI classification evaluations, wrap each evaluation resource as
{"dataset_id": "holdout-v2", "vertex_evaluation": {"metricsSchemaUri":
 "gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml",
 "metrics": {"auPrc": 0.84}}}. Use au_prc or au_roc as the metric.
Use an immutable dataset ID that includes the evaluation split and protocol.
The selected metric must be a score in [0, 1] where higher is better.
Exit status: 0 passes, 1 blocks promotion, 2 indicates invalid inputs.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path


def score(value, label):
    if not isinstance(value, Decimal) or not value.is_finite() or not 0 <= value <= 1:
        raise ValueError(f"{label} must be a finite number between 0 and 1")
    return value


def read_report(path, metric):
    report = json.loads(path.read_text(encoding="utf-8"), parse_int=Decimal, parse_float=Decimal)
    if not isinstance(report, dict):
        raise ValueError(f"{path}: expected a JSON object")
    dataset_id = report.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError(f"{path}: dataset_id must be a non-empty string")
    if "vertex_evaluation" in report:
        if "metrics" in report:
            raise ValueError(f"{path}: choose one evaluation report format")
        evaluation = report["vertex_evaluation"]
        schema = (
            "gs://google-cloud-aiplatform/schema/modelevaluation/"
            "classification_metrics_1.0.0.yaml"
        )
        if not isinstance(evaluation, dict) or evaluation.get("metricsSchemaUri") != schema:
            raise ValueError(f"{path}: expected a Vertex AI classification evaluation")
        vertex_metric = {"au_prc": "auPrc", "au_roc": "auRoc"}.get(metric)
        metrics = evaluation.get("metrics")
        if not isinstance(metrics, dict) or vertex_metric not in metrics:
            raise ValueError(f"{path}: missing Vertex AI metric {metric!r}")
        return dataset_id, score(metrics[vertex_metric], f"{path}: {metric}")

    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"{path}: missing metric {metric!r}")
    return dataset_id, score(metrics[metric], f"{path}: {metric}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--minimum", required=True, type=Decimal)
    parser.add_argument("--max-regression", required=True, type=Decimal)
    args = parser.parse_args()

    try:
        minimum = score(args.minimum, "minimum")
        allowed = score(args.max_regression, "max-regression")
        baseline_dataset, baseline = read_report(args.baseline, args.metric)
        candidate_dataset, candidate = read_report(args.candidate, args.metric)
        if baseline_dataset != candidate_dataset:
            raise ValueError("baseline and candidate dataset IDs must match")
        regression = baseline - candidate
    except (OSError, ValueError, InvalidOperation) as error:
        parser.error(str(error))

    passed = candidate >= minimum and regression <= allowed
    print(json.dumps({
        "passed": passed,
        "dataset_id": candidate_dataset,
        "metric": args.metric,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "regression": str(regression),
        "minimum": str(minimum),
        "max_regression": str(allowed),
    }))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

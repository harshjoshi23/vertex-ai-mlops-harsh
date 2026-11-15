# Model release gate

This Python 3.11+ CLI compares a model candidate with the current baseline before
promotion. It requires the candidate to meet a minimum score and stay within an
allowed regression. The comparison requires both reports to identify the same
evaluation dataset and protocol.

## Run the demo

From this directory, create two synthetic reports in a temporary directory.

```sh
reports=$(mktemp -d)
printf '%s\n' '{"dataset_id":"holdout-v2","metrics":{"au_prc":0.84}}' > "$reports/baseline.json"
printf '%s\n' '{"dataset_id":"holdout-v2","metrics":{"au_prc":0.82}}' > "$reports/candidate.json"
python3 check.py "$reports/baseline.json" "$reports/candidate.json" \
  --metric au_prc --minimum 0.80 --max-regression 0.02
```

This emits a JSON decision with `"passed": true`. Changing the candidate score
to 0.819 blocks promotion even though it still clears the minimum. Exit code 0
allows promotion, 1 blocks it on score, and 2 flags a bad input. Run the
regression suite with `python3 -m unittest discover -v`.

## Vertex AI evaluation inputs

Wrap each saved [Vertex AI classification evaluation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/Shared.Types/ClassificationEvaluationMetrics)
with the immutable dataset and protocol identifier from its pipeline manifest.
For example, a synthetic candidate report looks like this.

```json
{
  "dataset_id": "holdout-v2",
  "vertex_evaluation": {
    "metricsSchemaUri": "gs://google-cloud-aiplatform/schema/modelevaluation/classification_metrics_1.0.0.yaml",
    "metrics": {"auPrc": 0.82, "auRoc": 0.91}
  }
}
```

Create the baseline in the same format and run the demo command with the two
wrapped files. Use `--metric au_prc` or `--metric au_roc`. The gate checks the
schema and compares the dataset identifiers. The pipeline is responsible for
ensuring those identifiers describe the actual holdout and scoring protocol.
Keep the original evaluation resource names in the saved evaluations for audit.
The gate runs locally against saved files. Exporting the evaluations and
validating against a live Google Cloud project are follow-up work.

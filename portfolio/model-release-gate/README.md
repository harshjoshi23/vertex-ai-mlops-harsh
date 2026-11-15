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

A [Vertex AI classification evaluation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/rest/Shared.Types/ClassificationEvaluationMetrics)
publishes `auPrc` and `auRoc`. The pipeline should select one evaluation per
model, verify both used the same holdout and scoring protocol, and map
`metrics.auPrc` to `metrics.au_prc` in the gate report. Use the immutable
evaluation dataset and protocol identifier from the pipeline manifest for
`dataset_id`. The gate compares these identifiers, so the pipeline is
responsible for attaching them correctly. Keep the original evaluation resource
names alongside the reports for audit.

The demo uses local reports. Automated export from Vertex AI and cloud
validation are follow-up work.

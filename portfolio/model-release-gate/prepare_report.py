"""Build a gate report from a saved Vertex AI ListModelEvaluations response.

Supply the exact evaluation resource name and immutable dataset/protocol ID
from the same pipeline manifest. The dataset ID is supplied by the caller.
"""

import argparse
import json
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("response", type=Path)
    parser.add_argument("--evaluation", required=True, help="Full evaluation resource name")
    parser.add_argument("--dataset-id", required=True, help="Immutable dataset/protocol ID")
    args = parser.parse_args()

    try:
        response = json.loads(args.response.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or not isinstance(response.get("modelEvaluations"), list):
            raise ValueError("expected a modelEvaluations array")
        if not args.dataset_id.strip() or not args.evaluation.strip():
            raise ValueError("evaluation name and dataset ID must be non-empty")
        matches = [
            evaluation for evaluation in response["modelEvaluations"]
            if isinstance(evaluation, dict) and evaluation.get("name") == args.evaluation
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one matching evaluation in the saved response")
        report = {"dataset_id": args.dataset_id, "vertex_evaluation": matches[0]}
        output = json.dumps(report, allow_nan=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(output)


if __name__ == "__main__":
    main()

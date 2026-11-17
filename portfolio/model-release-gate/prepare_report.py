"""Build a gate report from a saved Vertex AI ListModelEvaluations response.

Supply the exact evaluation resource name and immutable dataset/protocol ID
from the same pipeline manifest. The dataset ID is supplied by the caller.
"""

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from check import unique_json_object

def dump_report(value):
    """Serialize JSON values without converting Decimal scores to floats."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{json.dumps(key)}: {dump_report(item)}" for key, item in value.items()
        ) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(map(dump_report, value)) + "]"
    return json.dumps(value, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("response", type=Path)
    parser.add_argument("--evaluation", required=True, help="Full evaluation resource name")
    parser.add_argument("--dataset-id", required=True, help="Immutable dataset/protocol ID")
    args = parser.parse_args()

    try:
        response = json.loads(
            args.response.read_text(encoding="utf-8"),
            parse_float=Decimal,
            object_pairs_hook=unique_json_object,
        )
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
        output = dump_report(report)
    except (OSError, ValueError, InvalidOperation) as error:
        parser.error(str(error))

    print(output)


if __name__ == "__main__":
    main()

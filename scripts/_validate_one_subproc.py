"""Single-problem validator worker, runnable as a subprocess for hard timeout.

Pattern mirrors scripts/_generate_one_hard.py: the parent spawns this with
`--job-json`, captures stdout's last line as JSON, and uses subprocess.run
timeout to enforce a hard per-problem deadline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the in-process worker function from validate_hard_set.
# Importing validate_hard_set runs only its top-level (no `main` is invoked).
from validate_hard_set import _validate_one  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job-json", type=str, required=True)
    args = p.parse_args()

    job = json.loads(args.job_json)
    result = _validate_one(job)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

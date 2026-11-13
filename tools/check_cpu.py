"""Run selected CPU contracts with private data and the feature validator's guards."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.validate_feature_cherry_picks import cpu_test_code, cpu_test_patterns


def run(patterns, *, root=ROOT, timeout=300):
    selected = cpu_test_patterns(root, patterns)
    # RF_DATA_DIR is set before any application import, including preset discovery.
    with tempfile.TemporaryDirectory(prefix="rf-cpu-") as private_data:
        environment = dict(os.environ)
        environment.update(
            RF_DATA_DIR=private_data,
            GRADIO_ANALYTICS_ENABLED="False",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        return subprocess.run(
            [sys.executable, "-c", cpu_test_code(selected)],
            cwd=root,
            env=environment,
            timeout=timeout,
            check=False,
        ).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Test filenames or globs inside tests/")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        return run(args.patterns, timeout=args.timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

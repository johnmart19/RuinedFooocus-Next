"""Build an exact-revision source ZIP without models, settings or environments."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ruinedfooocus.packaging.distribution import build_source_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    try:
        result = build_source_archive(ROOT, args.output, args.ref)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Source archive: {args.output} ({result['source_commit']})")
    print("Python/runtime and model weights are not bundled. Use the included launchers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

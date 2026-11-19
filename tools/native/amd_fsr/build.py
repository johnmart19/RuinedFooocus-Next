"""Build the AMD FSR still-image runtime ahead of first use.

Windows only; needs the Visual Studio C++ build tools. The application does this
itself the first time AMD FSR 2, 3 or 4 is requested. This command does the same
without starting it, for example to prepare an offline machine:

    python tools/native/amd_fsr/build.py [--sdk FOLDER] [--output DIR]

Without `--sdk` the pinned AMD FidelityFX SDK files are downloaded from AMD's
repository and verified by SHA256. With `--sdk`, an unpacked SDK folder is used
and nothing is downloaded. AMD's licence for the signed binaries requires its
notice to accompany redistributions; it is copied beside them.
"""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sdk", help="Folder of an unpacked AMD FSR SDK; default downloads it.")
    parser.add_argument("--output", help="Runtime folder; default is the application's.")
    options = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("The AMD FSR runtime needs Windows and DirectX 12.")
    sys.path.insert(0, str(ROOT))
    from ruinedfooocus.backends.amd_fsr import runtime
    from ruinedfooocus.features.image_processing.amd_fsr4 import runtime_directory

    try:
        sdk = options.sdk or runtime.fetch_sdk()
        output = runtime.build(sdk, options.output or runtime_directory())
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"AMD FSR runtime ready in {output}")


if __name__ == "__main__":
    main()

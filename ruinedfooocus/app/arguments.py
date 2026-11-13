import argparse
from ruinedfooocus.packaging.startup_options import LEGACY_SKIP_FLAG

parser = argparse.ArgumentParser()
parser.add_argument("--settings", type=str, default=None, help="Select setting")
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument(
    "--share",
    action="store_true",
    help="Unsupported: public tunnels are disabled; use authenticated --listen instead.",
)
parser.add_argument(
    "--auth",
    type=str,
    help="UI credentials username/password (or RF_UI_AUTH). Required for LAN access.",
)
parser.add_argument(
    "--guest",
    action="store_true",
    help="Restricted generation-only server; no existing gallery, settings, chat or owner API.",
)
parser.add_argument(
    "--listen",
    type=str,
    default=None,
    metavar="IP",
    nargs="?",
    const="0.0.0.0",
    help="Set the listen interface.",
)
parser.add_argument("--mcp", action="store_true", help="Start MCP server.")
parser.add_argument(
    "--api",
    action="store_true",
    help="Expose OpenAI-compatible chat/images and an OpenAPI image tool.",
)
parser.add_argument(
    "--api-key",
    default=None,
    help="API bearer key (or set RF_API_KEY). Required with --api when listening beyond localhost.",
)
parser.add_argument("--nobrowser", action="store_true", help="Do not launch in browser.")
devices = parser.add_mutually_exclusive_group()
devices.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")
devices.add_argument(
    "--cuda-device",
    default=None,
    metavar="DEVICE_IDS",
    help="Visible CUDA/HIP devices: comma-separated indices, or all.",
)
parser.add_argument(
    "--default-device",
    type=int,
    default=None,
    help="Prefer a physical GPU index (0-31), unless --cuda-device is supplied.",
)
parser.add_argument(
    "--oneapi-device-selector",
    default=None,
    help="Set the Intel oneAPI device selector before Torch startup.",
)
parser.add_argument("--offline", action="store_true", help="Disable startup updates and downloads.")
parser.add_argument(
    "--skip-dependency-install",
    action="store_true",
    help="Use installed packages without pip or native-runtime installation; allow source updates.",
)
parser.add_argument(
    "--rocm10",
    action="store_true",
    help="Install or repair ROCm 10 packages for a supported AMD GPU (Windows/Linux/WSL).",
)
parser.add_argument(
    "--cuda-nightly",
    action="store_true",
    help="Install compatible NVIDIA CUDA nightly wheels for this Python (Windows/Linux).",
)
parser.add_argument(
    "--legacy",
    action="store_true",
    help="Use compatible CUDA 12.6/12.4 wheels for older NVIDIA GPUs; keep modern CUDA separate.",
)
parser.add_argument(
    LEGACY_SKIP_FLAG, dest="skip_dependency_install", action="store_true", help=argparse.SUPPRESS
)
parser.add_argument(
    "--language", type=str, default=None, help="UI language (overrides the saved default)."
)
parser.add_argument("--clean-cache", action="store_true", help="Purge old cache before starting.")
parser.add_argument(
    "--preview-method",
    choices=("auto", "none", "latent2rgb", "taesd"),
    default="auto",
    help="Sampling preview decoder (default: auto).",
)

# ComfyUI compute arguments
from ruinedfooocus.backends.comfy.comfy_args import add_comfy_arguments

add_comfy_arguments(parser)

args = parser.parse_args()

from ruinedfooocus.backends.comfy.image_compute import load_compute, explicit_options
from ruinedfooocus.core.profile_store import selected_profile
import sys

options = explicit_options(parser, sys.argv[1:])
args.settings = selected_profile(args.settings)
profile = load_compute(args, parser, sys.argv[1:])
if "--language" not in options:
    args.language = profile.get("ui_language", profile.get("language", "en"))
if "--port" not in options and profile.get("startup_port") is not None:
    args.port = profile["startup_port"]
if "--nobrowser" not in options:
    args.nobrowser = not profile.get("startup_browser", True)
args.offline = args.offline or profile.get("startup_offline", False)
args.skip_dependency_install = args.skip_dependency_install or profile.get(
    "startup_skip_dependency_install", False
)
for option in ("api", "mcp", "listen"):
    if "--" + option not in options and profile.get("startup_" + option) is not None:
        setattr(args, option, profile["startup_" + option])
args._runtime_profile = args.settings

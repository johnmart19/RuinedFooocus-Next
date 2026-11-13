"""Qualified package plans shared by installation, diagnostics and packaging.

Wheel availability is checked separately for the running interpreter and OS.
These source-qualified plans do not certify execution on every listed device.
AMD companions follow AMD's release, rather than PyPI version ordering.
"""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class RuntimeBundle:
    torch: str
    vision: str
    index: str
    python_min: tuple = (3, 10)
    platforms: tuple = ("Windows", "Linux")
    architecture: str = "Wheel tags and selected device determine eligibility"
    qualification: str = "Publisher package compatibility; device acceptance required"
    python_max: tuple = (3, 14)


BUNDLES = MappingProxyType(
    {
        "cpu": RuntimeBundle(
            "2.15.0", "0.29.2", "https://download.pytorch.org/whl/cpu", python_min=(3, 11)
        ),
        "cu124": RuntimeBundle("2.6.0", "0.21.0", "https://download.pytorch.org/whl/cu124"),
        "cu126": RuntimeBundle(
            "2.14.1",
            "0.29.1",
            "https://download.pytorch.org/whl/cu126",
            architecture="Maxwell/Pascal/Volta and supported newer devices",
        ),
        "cu128": RuntimeBundle("2.11.0", "0.26.0", "https://download.pytorch.org/whl/cu128"),
        "cu130": RuntimeBundle(
            "2.14.1",
            "0.29.1",
            "https://download.pytorch.org/whl/cu130",
            architecture="Turing and newer",
        ),
        "cu132": RuntimeBundle(
            "2.15.0",
            "0.29.2",
            "https://download.pytorch.org/whl/cu132",
            python_min=(3, 11),
            architecture="Turing and newer",
        ),
        "cu134": RuntimeBundle(
            "2.15.0",
            "0.29.2",
            "https://download.pytorch.org/whl/cu134",
            python_min=(3, 11),
            architecture="Turing and newer; experimental Rubin support requires separate acceptance",
        ),
        # Intel Arc A/B-series and Core Ultra (Meteor/Arrow/Lunar/Panther Lake) GPUs.
        # Same torch line as the CPU bundle; the official XPU index carries matching
        # torchvision builds and pulls pytorch-triton-xpu itself.
        "xpu": RuntimeBundle(
            "2.15.0",
            "0.29.2",
            "https://download.pytorch.org/whl/xpu",
            python_min=(3, 11),
            architecture="Intel Arc and Core Ultra GPUs with a current Intel graphics driver",
        ),
        "rocm10.1": RuntimeBundle(
            "2.14.0+rocm10.1.0",
            "0.29.0a0+rocm10.1.0",
            "https://stable.repo.amd.com/rocm/whl-next/",
            python_min=(3, 11),
            architecture="AMD device extras; platform qualification checked by rocm_installer",
        ),
    }
)


def bundle_for(runtime, python_version=None):
    import sys

    version = sys.version_info[:2] if python_version is None else python_version
    # Keep deprecated Python 3.10 usable; 2.15 no longer publishes its wheels.
    if runtime in ("cpu", "cu132") and version < (3, 11):
        return RuntimeBundle(
            "2.14.1",
            "0.29.1",
            BUNDLES[runtime].index,
            architecture=BUNDLES[runtime].architecture,
        )
    return BUNDLES[runtime]


def python_bundle_supported(runtime, python_version=None, implementation="cpython"):
    """Publisher branch admission is separate from finding one matching wheel."""
    import sys

    version = sys.version_info[:2] if python_version is None else python_version
    if implementation != "cpython":
        return False
    if runtime == "directml":
        return (3, 10) <= version <= (3, 12)
    if runtime == "cu134":
        return (3, 11) <= version <= (3, 14)
    if runtime not in BUNDLES:
        return None
    bundle = bundle_for(runtime, version)
    return bundle.python_min <= version <= bundle.python_max

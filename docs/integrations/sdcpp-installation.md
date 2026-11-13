# Optional diffusion runtime installation

`ruinedfooocus/backends/sdcpp/artifacts.py` freezes publisher archives from
stable-diffusion.cpp revision `fc350be2ceeeb859bcc201ac435591ed7d46821d` and release
[`master-1068-fc350be`](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1068-fc350be),
reviewed 2026-11-09. It prepares a separate executable bundle without replacing
the app's Python, Torch or native ComfyUI runtime.

The installer verifies archive size/hash and each published payload, retains MIT
and GGML notices, rejects unsafe archive paths, and atomically publishes a new
directory. Existing bundles are verified and reused. Shared preparation owns
download progress, cancellation and file leases; installation takes a target
lock. Callers must enforce explicit installation, offline and platform policy.

The frozen matrix contains Windows x86-64 CPU/Vulkan and publisher Ubuntu 26.04
x86-64 CPU/Vulkan archives. Linux loader/CPU compatibility and WSL remain separate
acceptance requirements. No ARM, macOS or other GPU backend is implied. The pure
request contract travels with the installer; installation does not qualify a
device, model or workload, and imports perform no hardware discovery.

Validate with `python tools/check_cpu.py test_sdcpp_installer.py`. These tests use
inert local archives and test tampering, path rejection, license retention and
reuse; they do not download or execute a runtime.

# Linux GPU JIT prerequisites

The pinned Linux PyTorch/Triton stack compiles small host-side extensions when
some native GPU operations are first used. This applies to ROCm and CUDA; a
successful basic Torch matrix operation does not exercise every JIT path.

After package installation, `app/bootstrap.py` calls
`packaging/linux_gpu_prerequisites.py`. When Triton is installed for a GPU route,
it checks for GCC/Clang and the selected interpreter's `Python.h`. Missing
prerequisites on apt-based distributions are acquired with administrator access:
`gcc`, `libc6-dev`, and `python<major>.<minor>-dev` as needed. An explicit `CC`
is respected; a missing custom compiler requires correction instead of replacement.
Other package managers receive an actionable error. A custom Python may require
its own matching headers even after distro packages are installed.

Windows, CPU routes and offline/install-skip launches do not acquire this
toolchain. No model, compiler benchmark or GPU probe is run by the check.

`python tools/check_cpu.py test_linux_gpu_prerequisites.py test_startup_options.py`
checks installer behavior with mocked commands. The real Debian 13/Python 3.13
Janima first-use case originally failed inside Triton's `hip_utils` build because
no C compiler was installed. On 2026-11-10, the regular launcher acquired the
matching prerequisites, and the same real Janima prompt completed on ROCm in
about 33 seconds at the automatic 1024 × 1024 size. This does not certify other
GPUs, operating systems or every Triton kernel.

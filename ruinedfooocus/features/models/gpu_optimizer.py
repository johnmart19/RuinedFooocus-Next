"""Quantized copies of image checkpoints for the kernels this GPU runs natively.

The native runtime executes INT8 linear layers with dedicated kernels on AMD
(HIP) and NVIDIA devices. `create_int8_copy` writes a second checkpoint whose
large transformer linear weights are stored as INT8 with a per-row scale and a
ConvRot (grouped Hadamard) rotation, the layout those kernels read. Everything
else is copied byte for byte and the original file is never modified.

Quantization changes the numbers: the same seed gives a different image.
`benchmark_pair` measures both files on this machine so the choice is made on
evidence rather than on the format's name. Measured on an RX 7900 XTX with
ROCm 10.2 (2026-11-20): Anima 3988 MB to 2216 MB and about 7% faster; JANIMA
5572 MB to 3081 MB, about 6% faster and 2.4 GB less GPU memory in use.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import time

from ruinedfooocus.features.models.checkpoint_tools import (
    CHUNK_SIZE,
    _header,
    _publish,
    _stamp,
    _tensors,
    check_weights,
    format_size,
)

SUFFIX = ".int8.safetensors"
FP8_SUFFIX = ".fp8.safetensors"
PAIR_SUFFIXES = (SUFFIX, FP8_SUFFIX, ".bf16.safetensors", ".fp16.safetensors")
GROUP_SIZE = 256
MIN_FEATURES = 1024
RECORD = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP_SIZE}
FP8_RECORD = {"format": "float8_e4m3fn"}
FLOAT_DTYPES = {"F16": "float16", "BF16": "bfloat16", "F32": "float32"}
# Only transformer blocks: their attention and MLP projections carry the weight.
BLOCK = re.compile(r"(?:^|\.)(?:[a-z_]*blocks|layers)\.(?:[a-z_]+\.)?\d+\.")
# Embedders, modulation and output layers stay at full precision, as in published files.
KEPT = ("adaln", "modulation", "embed", "emb_in", "final", "norm", "proj_out", "time_")
OTHER_MODELS = ("text_encoders.", "cond_stage_model.", "conditioner.", "first_stage_model.", "vae.")
BENCHMARK_PROMPT = (
    "A lighthouse on a rocky coast under a cloudy sky, a person in a red coat "
    "holding a lantern, detailed illustration."
)


def eligible(name, tensor):
    """Whether one tensor is a transformer linear weight the INT8 kernels accept."""
    shape = tensor["shape"]
    return (
        name.endswith(".weight")
        and tensor["dtype"] in FLOAT_DTYPES
        and len(shape) == 2
        and min(shape) >= MIN_FEATURES
        and shape[1] % GROUP_SIZE == 0
        and not name.startswith(OTHER_MODELS)
        and BLOCK.search(name) is not None
        and not any(part in name for part in KEPT)
    )


def plan(path):
    """Header-only decision: which layers would change and what the copy would weigh."""
    path = Path(path)
    header, _ = _header(path)
    tensors = _tensors(header)
    names = {name for name, _ in tensors}
    if any(name.endswith(".comfy_quant") for name in names) or "_quantization_metadata" in (
        header.get("__metadata__") or {}
    ):
        raise ValueError("This checkpoint is already quantized. No copy created.")
    if any("input_blocks." in name for name in names):
        raise ValueError(
            "INT8 copies are for transformer image models (Anima, Flux, Qwen and similar). "
            "This is a UNet checkpoint."
        )
    layers = [name for name, tensor in tensors if eligible(name, tensor)]
    if not layers:
        raise ValueError("No transformer linear layers to quantize. No copy created.")
    size = lambda tensor: tensor["data_offsets"][1] - tensor["data_offsets"][0]  # noqa: E731
    before = sum(size(tensor) for _, tensor in tensors)
    record = len(json.dumps(RECORD).encode("utf-8"))
    after = before
    for name in layers:
        tensor = header[name]
        rows, columns = tensor["shape"]
        after += rows * columns + rows * 4 + record - size(tensor)
    return dict(layers=layers, bytes_before=before, bytes_after=after)


def _quantize(raw, tensor):
    import torch

    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    weight = torch.frombuffer(bytearray(raw), dtype=getattr(torch, FLOAT_DTYPES[tensor["dtype"]]))
    weight = weight.reshape(tensor["shape"]).float()
    # On the CPU: no shared GPU state is touched while the image worker may be busy.
    data, params = TensorWiseINT8Layout.quantize(
        weight, per_channel=True, convrot=True, convrot_groupsize=GROUP_SIZE
    )
    scale = params.scale.float().reshape(-1, 1)
    if tuple(data.shape) != tuple(weight.shape) or scale.shape[0] != weight.shape[0]:
        raise ValueError("The INT8 kernel returned an unexpected layout. Conversion stopped.")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("A layer has no finite INT8 scale. Conversion stopped.")
    return data.contiguous().numpy().tobytes(), scale.contiguous().numpy().tobytes()


def create_int8_copy(
    path, cache_dir, model_info=None, preview=None, progress=lambda fraction, description: None
):
    """Write `<name>.int8.safetensors` beside the source; never replaces a file."""
    path = Path(path)
    planned = plan(path)
    destination = path.with_name(path.stem + SUFFIX)
    cache_dir = Path(cache_dir)
    cached = cache_dir / destination.name
    metadata_destination = cached.with_suffix(".json")
    preview = Path(preview) if preview else None
    preview_destination = cached.with_suffix(preview.suffix) if preview else None
    for target in (destination, metadata_destination, preview_destination):
        if target and target.exists():
            raise FileExistsError(f"Already exists: {target.name}")
    if shutil.disk_usage(path.parent).free < planned["bytes_after"] + CHUNK_SIZE:
        raise ValueError("Not enough free disk space for the INT8 copy.")
    before = _stamp(path)
    checked = check_weights(path, progress)
    if checked["invalid"]:
        raise ValueError("Checkpoint contains NaN/Inf weights. Conversion stopped.")
    header, source_offset = _header(path)
    chosen = set(planned["layers"])
    record = json.dumps(RECORD).encode("utf-8")
    converted, position = {}, 0

    def place(name, dtype, shape, size):
        nonlocal position
        converted[name] = dict(dtype=dtype, shape=shape, data_offsets=[position, position + size])
        position += size

    for name, tensor in _tensors(header):
        if name in chosen:
            rows, columns = tensor["shape"]
            prefix = name[: -len("weight")]
            place(name, "I8", [rows, columns], rows * columns)
            place(prefix + "weight_scale", "F32", [rows, 1], rows * 4)
            place(prefix + "comfy_quant", "U8", [len(record)], len(record))
        else:
            start, end = tensor["data_offsets"]
            place(name, tensor["dtype"], tensor["shape"], end - start)
    converted["__metadata__"] = (header.get("__metadata__") or {}) | {
        "rf.source_sha256": checked["sha256"],
        "rf.source_name": path.name,
        "rf.conversion": f"INT8 ConvRot with per-row scale for {len(chosen)} linear layers; "
        "other tensors preserved",
    }
    encoded = json.dumps(converted, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    started = time.monotonic()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory,
        TemporaryDirectory(prefix=".checkpoint-", dir=cache_dir) as cache_temporary,
    ):
        temporary = Path(directory) / "weights.tmp"
        digest = hashlib.sha256()
        with path.open("rb") as source, temporary.open("wb") as output:

            def write(raw):
                output.write(raw)
                digest.update(raw)

            write(len(encoded).to_bytes(8, "little") + encoded)
            for name, tensor in _tensors(header):
                start, end = tensor["data_offsets"]
                source.seek(source_offset + start)
                if name in chosen:
                    raw = source.read(end - start)
                    if len(raw) != end - start:
                        raise ValueError("Checkpoint was truncated during conversion.")
                    data, scale = _quantize(raw, tensor)
                    write(data)
                    write(scale)
                    write(record)
                else:
                    remaining = end - start
                    while remaining:
                        raw = source.read(min(CHUNK_SIZE, remaining))
                        if not raw:
                            raise ValueError("Checkpoint was truncated during conversion.")
                        remaining -= len(raw)
                        write(raw)
                progress(source.tell() / before[0], "Writing INT8 copy")
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != 8 + len(encoded) + position:
            raise ValueError("File verification failed. Conversion stopped.")
        from safetensors import safe_open

        with safe_open(temporary, framework="pt", device="cpu"):
            pass
        if _stamp(path) != before:
            raise ValueError("Checkpoint changed during conversion; try again.")
        info = dict(model_info or {})
        info.update(
            baseModel=info.get("baseModel") or "Unknown",
            files=[{"name": destination.name, "hashes": {"SHA256": digest.hexdigest()}}],
            rf_source={"name": path.name, "sha256": checked["sha256"]},
        )
        metadata = Path(cache_temporary) / "metadata.tmp"
        metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")
        publish = [(metadata, metadata_destination)]
        if preview:
            staged_preview = Path(cache_temporary) / "preview.tmp"
            shutil.copyfile(preview, staged_preview)
            publish.append((staged_preview, preview_destination))
        publish.append((temporary, destination))
        created = []
        try:
            for source, target in publish:
                _publish(source, target)
                created.append(target)
        except OSError:
            for target in reversed(created):
                target.unlink()
            raise
    saved = planned["bytes_before"] - planned["bytes_after"]
    return dict(
        path=destination,
        preview=preview_destination,
        layers=len(chosen),
        message=f"Created {destination.name} in {time.monotonic() - started:.0f} s. "
        f"{len(chosen)} linear layers stored as INT8, saving {format_size(saved)}. "
        "Original retained. The same seed gives a different image with this copy; "
        "use Benchmark pair to compare speed on this GPU.",
    )


def _quantize_fp8(raw, tensor):
    import torch

    weight = torch.frombuffer(bytearray(raw), dtype=getattr(torch, FLOAT_DTYPES[tensor["dtype"]]))
    weight = weight.reshape(tensor["shape"]).float()
    try:
        from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

        data, params = TensorCoreFP8Layout.quantize(weight, dtype=torch.float8_e4m3fn)
        scale = params.scale.float().reshape(())
    except (ImportError, RuntimeError):
        fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
        max_abs = weight.abs().max()
        scale = (max_abs / fp8_max).clamp(min=1e-12).float().reshape(())
        data = (weight / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    if tuple(data.shape) != tuple(weight.shape):
        raise ValueError("The FP8 kernel returned an unexpected layout. Conversion stopped.")
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("A layer has no finite FP8 scale. Conversion stopped.")
    return data.view(
        torch.uint8
    ).contiguous().numpy().tobytes(), scale.contiguous().numpy().tobytes()


def create_fp8_copy(
    path, cache_dir, model_info=None, preview=None, progress=lambda fraction, description: None
):
    """Write `<name>.fp8.safetensors` beside the source using ComfyUI float8_e4m3fn layout."""
    path = Path(path)
    planned = plan(path)
    destination = path.with_name(path.stem + FP8_SUFFIX)
    cache_dir = Path(cache_dir)
    cached = cache_dir / destination.name
    metadata_destination = cached.with_suffix(".json")
    preview = Path(preview) if preview else None
    preview_destination = cached.with_suffix(preview.suffix) if preview else None
    for target in (destination, metadata_destination, preview_destination):
        if target and target.exists():
            raise FileExistsError(f"Already exists: {target.name}")
    if shutil.disk_usage(path.parent).free < planned["bytes_after"] + CHUNK_SIZE:
        raise ValueError("Not enough free disk space for the FP8 copy.")
    before = _stamp(path)
    checked = check_weights(path, progress)
    if checked["invalid"]:
        raise ValueError("Checkpoint contains NaN/Inf weights. Conversion stopped.")
    header, source_offset = _header(path)
    chosen = set(planned["layers"])
    record = json.dumps(FP8_RECORD).encode("utf-8")
    converted, position = {}, 0

    def place(name, dtype, shape, size):
        nonlocal position
        converted[name] = dict(dtype=dtype, shape=shape, data_offsets=[position, position + size])
        position += size

    for name, tensor in _tensors(header):
        if name in chosen:
            rows, columns = tensor["shape"]
            prefix = name[: -len("weight")]
            place(name, "F8_E4M3", [rows, columns], rows * columns)
            place(prefix + "weight_scale", "F32", [], 4)
            place(prefix + "comfy_quant", "U8", [len(record)], len(record))
        else:
            start, end = tensor["data_offsets"]
            place(name, tensor["dtype"], tensor["shape"], end - start)
    converted["__metadata__"] = (header.get("__metadata__") or {}) | {
        "rf.source_sha256": checked["sha256"],
        "rf.source_name": path.name,
        "rf.conversion": f"FP8 E4M3FN with per-tensor scale for {len(chosen)} linear layers; "
        "other tensors preserved",
    }
    encoded = json.dumps(converted, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    started = time.monotonic()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory,
        TemporaryDirectory(prefix=".checkpoint-", dir=cache_dir) as cache_temporary,
    ):
        temporary = Path(directory) / "weights.tmp"
        digest = hashlib.sha256()
        with path.open("rb") as source, temporary.open("wb") as output:

            def write(raw):
                output.write(raw)
                digest.update(raw)

            write(len(encoded).to_bytes(8, "little") + encoded)
            for name, tensor in _tensors(header):
                start, end = tensor["data_offsets"]
                source.seek(source_offset + start)
                if name in chosen:
                    raw = source.read(end - start)
                    if len(raw) != end - start:
                        raise ValueError("Checkpoint was truncated during conversion.")
                    data, scale = _quantize_fp8(raw, tensor)
                    write(data)
                    write(scale)
                    write(record)
                else:
                    remaining = end - start
                    while remaining:
                        raw = source.read(min(CHUNK_SIZE, remaining))
                        if not raw:
                            raise ValueError("Checkpoint was truncated during conversion.")
                        remaining -= len(raw)
                        write(raw)
                progress(source.tell() / before[0], "Writing FP8 copy")
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size != 8 + len(encoded) + position:
            raise ValueError("File verification failed. Conversion stopped.")
        from safetensors import safe_open

        with safe_open(temporary, framework="pt", device="cpu"):
            pass
        if _stamp(path) != before:
            raise ValueError("Checkpoint changed during conversion; try again.")
        info = dict(model_info or {})
        info.update(
            baseModel=info.get("baseModel") or "Unknown",
            files=[{"name": destination.name, "hashes": {"SHA256": digest.hexdigest()}}],
            rf_source={"name": path.name, "sha256": checked["sha256"]},
        )
        metadata = Path(cache_temporary) / "metadata.tmp"
        metadata.write_text(json.dumps(info, indent=2), encoding="utf-8")
        publish = [(metadata, metadata_destination)]
        if preview:
            staged_preview = Path(cache_temporary) / "preview.tmp"
            shutil.copyfile(preview, staged_preview)
            publish.append((staged_preview, preview_destination))
        publish.append((temporary, destination))
        created = []
        try:
            for source, target in publish:
                _publish(source, target)
                created.append(target)
        except OSError:
            for target in reversed(created):
                target.unlink()
            raise
    saved = planned["bytes_before"] - (8 + len(encoded) + position)
    return dict(
        path=destination,
        preview=preview_destination,
        layers=len(chosen),
        message=f"Created {destination.name} in {time.monotonic() - started:.0f} s. "
        f"{len(chosen)} linear layers stored as FP8 E4M3FN, saving {format_size(max(0, saved))}. "
        "Original retained. Use Benchmark pair to compare speed on this GPU.",
    )


def pair(name, names):
    """(original, optimized copy) for a selected checkpoint when both are installed."""
    current = Path(name)
    for suffix in PAIR_SUFFIXES:
        if current.name.endswith(suffix):
            stem = current.name[: -len(suffix)]
            originals = [
                other
                for other in names
                if Path(other).parent == current.parent
                and Path(other).stem == stem
                and not any(Path(other).name.endswith(s) for s in PAIR_SUFFIXES)
            ]
            return (originals[0], name) if originals else None
    for suffix in PAIR_SUFFIXES:
        copy = str(current.with_name(current.stem + suffix)).replace("\\", "/")
        matches = [other for other in names if other.replace("\\", "/") == copy]
        if matches:
            return (name, matches[0])
    return None


def device_memory_used():
    """GPU memory in use on the image device in GiB, or None without a GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 2**30
    except Exception:
        return None


def benchmark_pair(original, copy, *, size="1024x1024", runs=2, generate=None, memory=None):
    """Generate with both files and report the last run of each, in seconds.

    The first run of a model includes loading it, so only the last run is compared.
    GPU memory is read after each model's last run, while it is still loaded; the
    reading covers everything on the device, so compare the two, not the absolute.
    """
    memory = device_memory_used if memory is None else memory
    if generate is None:
        from ruinedfooocus.app import api_runtime

        def generate(name):
            resolved, selected = api_runtime.resolve_model("image/" + name, "image")
            payload = dict(prompt=BENCHMARK_PROMPT, model=resolved, n=1, seed=1234, size=size)
            api_runtime.wait_for_result(api_runtime.images(payload, selected))

    timings, used = {}, {}
    for name in (original, copy):
        for _ in range(runs):
            started = time.perf_counter()
            generate(name)
            timings[name] = time.perf_counter() - started
        used[name] = memory()
    first, second = timings[original], timings[copy]
    change = (first - second) / first * 100
    label = (
        "INT8"
        if copy.endswith(SUFFIX)
        else "FP8"
        if copy.endswith(FP8_SUFFIX)
        else "BF16"
        if copy.endswith(".bf16.safetensors")
        else "FP16"
        if copy.endswith(".fp16.safetensors")
        else "optimized"
    )
    verdict = (
        f"The {label} copy is {change:.0f}% faster."
        if change >= 2
        else f"The {label} copy is {-change:.0f}% slower."
        if change <= -2
        else "No measurable speed difference."
    )
    memory_note = ""
    if used[original] is not None and used[copy] is not None:
        memory_note = (
            f" GPU memory in use: {used[original]:.1f} GiB with the original, "
            f"{used[copy]:.1f} GiB with the copy."
        )
    return dict(
        seconds=timings,
        memory_gib=used,
        change_percent=change,
        message=f"{size}, run {runs} of {runs} each: {Path(original).name} {first:.1f} s, "
        f"{Path(copy).name} {second:.1f} s. {verdict}{memory_note} "
        "Both images are in the Gallery for a quality comparison.",
    )

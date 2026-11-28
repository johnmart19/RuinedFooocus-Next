"""Explicit source verification, safetensors inspection and non-destructive copies."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from safetensors import safe_open
from ruinedfooocus.core.model_formats import is_safetensors
from ruinedfooocus.core.config_io import save_json
from ruinedfooocus.features.models.model_identity import (
    cached_file_sha256,
    file_signature,
    is_native_checkpoint_source,
)

CHUNK_SIZE = 16 * 1024 * 1024


def verify_checkpoint_source(path, cache_dir, model_info=None):
    """Explicit source verification, not tensor inspection or native model execution."""
    from ruinedfooocus.features.models.model_registry import load_registry

    path, cache_dir = Path(path), Path(cache_dir)
    signature = file_signature(path)
    checksum = cached_file_sha256(path, cache_dir.parent / "hashes").lower()
    matches = [
        row
        for row in load_registry().entries()
        if is_native_checkpoint_source(row)
        and row["artifact"]["sha256"] == checksum
        and row["artifact"]["bytes"] == signature[1]
    ]
    families = {row["native_family"] for row in matches}
    if len(families) != 1:
        raise ValueError(
            "No reviewed native family matches this source checksum and size. "
            "No verified family was assigned; no tensors were loaded or scanned."
        )
    if file_signature(path) != signature:
        raise ValueError("Checkpoint changed during source verification; inspect again.")
    family = families.pop()
    info = dict(model_info or {})
    info.update(
        baseModel=family,
        _rf_local_sha256=checksum,
        _rf_file_signature=signature,
        _rf_verified_native_source={
            "artifact": matches[0]["id"],
            "family": family,
            "sha256": checksum,
        },
    )
    save_json((cache_dir / path.name).with_suffix(".json"), info)
    return dict(family=family, sha256=checksum, bytes=signature[1])


def _header(path):
    path = Path(path)
    if not is_safetensors(path):
        raise ValueError("Select a .safetensors or .sft checkpoint.")
    # Validate offsets, shapes and file length using the format's own reader.
    with safe_open(path, framework="pt", device="cpu"):
        pass
    with path.open("rb") as source:
        size = int.from_bytes(source.read(8), "little")
        if size > 16 * 1024 * 1024:
            raise ValueError("Checkpoint header is too large to inspect.")
        return json.loads(source.read(size)), size + 8


def inspect_checkpoint(path):
    path = Path(path)
    header, _ = _header(path)
    sizes = Counter()
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    for tensor in tensors:
        start, end = tensor["data_offsets"]
        sizes[tensor["dtype"]] += end - start
    return dict(
        name=path.name,
        tensors=len(tensors),
        bytes=path.stat().st_size,
        dtypes=dict(sizes),
        saving_bytes=sizes["F32"] // 2,
        previous_repair=header.get("__metadata__", {}).get("donor_repair") == "Applied",
    )


def _float_values(raw, dtype):
    if dtype in ("F16", "F32", "F64"):
        return np.frombuffer(raw, dtype={"F16": "<f2", "F32": "<f4", "F64": "<f8"}[dtype])
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype.startswith("F"):
        import torch

        names = {
            "F8_E4M3": "float8_e4m3fn",
            "F8_E5M2": "float8_e5m2",
            "F8_E4M3FNUZ": "float8_e4m3fnuz",
            "F8_E5M2FNUZ": "float8_e5m2fnuz",
            "F8_E8M0": "float8_e8m0fnu",
        }
        target = getattr(torch, names.get(dtype, ""), None)
        if target is None:
            raise ValueError(f"Weight checking is not supported for {dtype}.")
        return torch.frombuffer(bytearray(raw), dtype=target).float().numpy()
    return None


def _tensors(header):
    return sorted(
        ((k, v) for k, v in header.items() if k != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    )


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def check_weights(path, progress=lambda fraction, description: None):
    path = Path(path)
    before = _stamp(path)
    header, offset = _header(path)
    digest = hashlib.sha256()
    invalid = []
    zeros = 0
    with path.open("rb") as source:
        digest.update(source.read(offset))
        for name, tensor in _tensors(header):
            start, end = tensor["data_offsets"]
            remaining = end - start
            source.seek(offset + start)
            nan = inf = 0
            all_zero = remaining > 0
            while remaining:
                raw = source.read(min(CHUNK_SIZE, remaining))
                if not raw:
                    raise ValueError("Checkpoint was truncated during checking.")
                digest.update(raw)
                remaining -= len(raw)
                values = _float_values(raw, tensor["dtype"])
                if values is not None:
                    nan += int(np.count_nonzero(np.isnan(values)))
                    inf += int(np.count_nonzero(np.isinf(values)))
                    all_zero &= not np.any(values)
                else:
                    all_zero = False
                progress(source.tell() / before[0], "Checking weights")
            if nan or inf:
                invalid.append(dict(name=name, nan=nan, inf=inf))
            zeros += int(all_zero)
    if _stamp(path) != before:
        raise ValueError("Checkpoint changed during checking; try again.")
    return dict(sha256=digest.hexdigest(), invalid=invalid, zero_tensors=zeros)


def _publish(source, target):
    """Publish a finished file without ever replacing an existing one."""
    try:
        # Atomic and no-overwrite on NTFS and Linux filesystems.
        os.link(source, target)
        return
    except FileExistsError:
        raise
    except OSError:
        # exFAT/FAT drives and some network shares have no hard links.
        if os.name == "nt":
            # Unlike POSIX rename, Windows rename refuses to replace a target.
            os.rename(source, target)
            return
    # Without hard links, POSIX rename could overwrite a concurrently created
    # target. Exclusive creation keeps the no-overwrite guarantee on these drives.
    destination = target.open("xb")
    try:
        with destination, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, destination, CHUNK_SIZE)
            destination.flush()
            os.fsync(destination.fileno())
        source.unlink()
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _create_reduced_copy(
    path,
    cache_dir,
    target_dtype,
    model_info=None,
    preview=None,
    progress=lambda fraction, description: None,
):
    label = "BF16" if target_dtype == "BF16" else "FP16"
    suffix = ".bf16.safetensors" if target_dtype == "BF16" else ".fp16.safetensors"
    path = Path(path)
    report = inspect_checkpoint(path)
    if not report["saving_bytes"]:
        return dict(path=None, message="No FP32 tensors to reduce. No copy created.")
    if report["saving_bytes"] < sum(report["dtypes"].values()) * 0.01:
        return dict(path=None, message=f"{label} would save less than 1%. No copy created.")
    destination = path.with_name(path.stem + suffix)
    cache_dir = Path(cache_dir)
    cached = cache_dir / destination.name
    metadata_destination = cached.with_suffix(".json")
    preview = Path(preview) if preview else None
    preview_destination = cached.with_suffix(preview.suffix) if preview else None
    for target in (destination, metadata_destination, preview_destination):
        if target and target.exists():
            raise FileExistsError(f"Already exists: {target.name}")
    needed = report["bytes"] - report["saving_bytes"] + CHUNK_SIZE
    if shutil.disk_usage(path.parent).free < needed:
        raise ValueError(f"Not enough free disk space for the {label} copy.")
    before = _stamp(path)
    checked = check_weights(path, progress)
    if checked["invalid"]:
        raise ValueError("Checkpoint contains NaN/Inf weights. Conversion stopped.")
    header, source_offset = _header(path)
    converted = {}
    position = 0
    for name, tensor in _tensors(header):
        start, end = tensor["data_offsets"]
        size = (end - start) // 2 if tensor["dtype"] == "F32" else end - start
        converted[name] = tensor | {"data_offsets": [position, position + size]}
        if tensor["dtype"] == "F32":
            converted[name]["dtype"] = target_dtype
        position += size
    converted["__metadata__"] = header.get("__metadata__", {}) | {
        "rf.source_sha256": checked["sha256"],
        "rf.source_name": path.name,
        "rf.conversion": f"FP32 to {label}; other dtypes preserved",
    }
    encoded = json.dumps(converted, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    underflow = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".checkpoint-", dir=path.parent) as directory,
        TemporaryDirectory(prefix=".checkpoint-", dir=cache_dir) as cache_temporary,
    ):
        temporary = Path(directory) / "weights.tmp"
        digest = hashlib.sha256()
        with path.open("rb") as source, temporary.open("wb") as output:
            prefix = len(encoded).to_bytes(8, "little") + encoded
            output.write(prefix)
            digest.update(prefix)
            for name, tensor in _tensors(header):
                start, end = tensor["data_offsets"]
                source.seek(source_offset + start)
                remaining = end - start
                while remaining:
                    raw = source.read(min(CHUNK_SIZE, remaining))
                    if not raw:
                        raise ValueError("Checkpoint was truncated during conversion.")
                    remaining -= len(raw)
                    if tensor["dtype"] == "F32":
                        values = np.frombuffer(raw, dtype="<f4")
                        if target_dtype == "BF16":
                            bits = np.frombuffer(raw, dtype="<u4")
                            rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
                            bf16_u16 = ((bits + rounding) >> np.uint32(16)).astype("<u2")
                            underflow += int(
                                np.count_nonzero(
                                    (values != 0) & ((bf16_u16 & np.uint16(0x7FFF)) == 0)
                                )
                            )
                            raw = bf16_u16.tobytes()
                        else:
                            with np.errstate(over="ignore", under="ignore"):
                                reduced = values.astype("<f2")
                            if not np.isfinite(reduced).all():
                                raise ValueError(f"FP16 overflow in {name}. Conversion stopped.")
                            underflow += int(np.count_nonzero((values != 0) & (reduced == 0)))
                            raw = reduced.tobytes()
                    output.write(raw)
                    digest.update(raw)
                    progress(source.tell() / before[0], f"Writing {label} copy")
            output.flush()
            os.fsync(output.fileno())
        with safe_open(temporary, framework="pt", device="cpu"):
            pass
        with temporary.open("rb") as output:
            verified = hashlib.sha256()
            for chunk in iter(lambda: output.read(CHUNK_SIZE), b""):
                verified.update(chunk)
            output_hash = verified.hexdigest()
        if output_hash != digest.hexdigest() or _stamp(path) != before:
            raise ValueError("File verification failed. Conversion stopped.")
        info = dict(model_info or {})
        info.update(
            baseModel=info.get("baseModel") or "Unknown",
            files=[{"name": destination.name, "hashes": {"SHA256": output_hash}}],
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
    return dict(
        path=destination,
        preview=preview_destination,
        underflow=underflow,
        message=f"Created {destination.name}. Saved {format_size(report['saving_bytes'])}. "
        f"Values rounded to zero: {underflow}. Original retained.",
    )


def create_fp16_copy(
    path, cache_dir, model_info=None, preview=None, progress=lambda fraction, description: None
):
    return _create_reduced_copy(path, cache_dir, "F16", model_info, preview, progress)


def create_bf16_copy(
    path, cache_dir, model_info=None, preview=None, progress=lambda fraction, description: None
):
    return _create_reduced_copy(path, cache_dir, "BF16", model_info, preview, progress)


def format_size(size):
    """Binary units that stay readable from small LoRAs to large checkpoints."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024


def format_report(report, family="Unknown", checked=None, *, quant_summary=None, paired_copy=None):
    dtypes = ", ".join(f"{name}: {format_size(size)}" for name, size in report["dtypes"].items())
    payload = sum(report["dtypes"].values())
    fp16_useful = bool(payload and report["saving_bytes"] >= payload * 0.01)
    lines = [
        report["name"],
        f"Model: {family} | {format_size(report['bytes'])} | {report['tensors']} tensors",
        f"Precision: {dtypes}",
        "",
        "Optimization options:",
    ]
    if fp16_useful:
        lines.append(
            f"• FP16 / BF16 copy: Recommended — saves ~{format_size(report['saving_bytes'])} from FP32 weights (BF16 avoids overflow on modern transformers)."
        )
    else:
        lines.append(
            "• FP16 / BF16 copy: Not needed (already reduced precision; no significant FP32 weights)."
        )
    if quant_summary is not None:
        if quant_summary.get("eligible"):
            saved = max(0, quant_summary["bytes_before"] - quant_summary["bytes_after"])
            lines.append(
                f"• FP8 / INT8 copy: Available — {quant_summary['layers']} transformer layers, "
                f"{format_size(quant_summary['bytes_before'])} → ~{format_size(quant_summary['bytes_after'])} "
                f"(saves ~{format_size(saved)})."
            )
        else:
            lines.append(
                f"• FP8 / INT8 copy: Not applicable ({quant_summary.get('reason', 'not eligible')})."
            )
    if paired_copy:
        lines.append(f"• Benchmark pair: Ready to compare against {Path(paired_copy).name}.")
    else:
        lines.append("• Benchmark pair: Create an optimized copy first to compare speed and VRAM.")
    if report["previous_repair"]:
        lines.append(
            "Note: Previous optimizer reports donor-replaced weights; this cannot verify their origin."
        )
    lines.append("")
    if checked is None:
        lines.append(
            "Weight integrity: Not scanned yet (click Check weights to scan every tensor for NaN/Inf and compute SHA-256)."
        )
    else:
        if checked["invalid"]:
            lines.append(
                f"Weight integrity: CORRUPTED — {len(checked['invalid'])} tensor(s) contain NaN/Inf! "
                f"All-zero tensors: {checked['zero_tensors']}."
            )
            lines.extend(
                f"  - {item['name']}: {item['nan']} NaN, {item['inf']} Inf"
                for item in checked["invalid"][:8]
            )
        else:
            lines.append(
                f"Weight integrity: PASS — all {report['tensors']} tensors are finite (0 NaN/Inf). "
                f"All-zero tensors: {checked['zero_tensors']} (may be intentional)."
            )
        lines.append(f"SHA-256 (local): {checked['sha256']}")
    return "\n".join(lines)

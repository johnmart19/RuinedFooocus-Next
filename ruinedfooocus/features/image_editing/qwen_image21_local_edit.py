"""Localized Qwen Image 2.1 editing by painting on its first reference."""

from PIL import Image, ImageColor

from ruinedfooocus.features.image_editing.inpaint_utils import prepare_mask


PAINT_COLOR = "#ff0000"
LOCAL_EDIT_MODES = ("paint", "mask")


def local_edit_mode(value):
    if value not in LOCAL_EDIT_MODES:
        raise ValueError("Choose Paint guide or Mask for Qwen Image 2.1 localized editing.")
    return value


def replay_values(metadata):
    """Restore only the edit mode; references and painted pixels are never metadata."""
    value = metadata.get("qwen_local_edit_mode")
    if value is None:
        frame = metadata.get("image_geometry")
        value = frame.get("qwen_local_edit_mode") if isinstance(frame, dict) else None
    return {} if value is None else {"qwen_local_edit_mode": local_edit_mode(value)}


def _mix_rgba(source, edited, coverage):
    """Blend premultiplied colors while preserving both exact endpoints."""
    import torch

    coverage = coverage.to(source)
    source_weight = source[..., 3:] * (1 - coverage)
    edited_weight = edited[..., 3:] * coverage
    alpha = source_weight + edited_weight
    rgb = (source[..., :3] * source_weight + edited[..., :3] * edited_weight) / alpha.clamp_min(
        torch.finfo(alpha.dtype).tiny
    )
    mixed = torch.cat((rgb, alpha), dim=-1)
    return torch.where(coverage == 0, source, torch.where(coverage == 1, edited, mixed))


def prepare_paint_guide(pixels, mask, frame):
    """Render the painted coverage as red in image_1, including on transparent pixels."""
    coverage = prepare_mask(mask, frame, Image.Resampling.BILINEAR).unsqueeze(-1)
    if not coverage.any():
        raise ValueError("The selected framing crops out the entire painted area.")
    red = pixels.new_tensor(ImageColor.getcolor(PAINT_COLOR, "RGBA")) / 255
    return _mix_rgba(pixels, red, coverage), coverage


def composite_local_edit(decoded, source, coverage):
    if decoded.ndim != 4 or decoded.shape[1:] != source.shape[1:]:
        raise ValueError("The Qwen Image 2.1 edit must decode to the aligned RGBA source canvas.")
    decoded = decoded.float()
    return _mix_rgba(source.to(decoded), decoded, coverage)

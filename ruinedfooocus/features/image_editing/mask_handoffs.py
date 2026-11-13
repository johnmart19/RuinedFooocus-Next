"""Owned, exact-frame selection masks for the existing native inpaint route."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResultMaskHandoff:
    session_id: str
    revision: int
    source_id: str
    mask_id: str
    source_frame_id: str
    mask_frame_id: str
    choice: str
    instance_index: int | None
    instance_count: int
    inverted: bool

    def metadata(self):
        return dict(
            asdict(self),
            purpose="selection-mask",
            channel="L",
            convention="white-selects; black-preserves; gray-fractional",
            alpha_policy="selection is independent of source alpha",
        )


def _refs(store, session_id, revision, source_id, mask_id, owner):
    snapshot = store.snapshot(session_id, owner=owner)
    if snapshot.revision != revision or snapshot.current.artifact_id != source_id:
        raise ValueError("The image changed. Select a mask for the current revision again.")
    source = store.artifact(session_id, source_id, revision=revision, owner=owner)
    mask = store.artifact(session_id, mask_id, revision=revision, owner=owner)
    if source.kind != "image" or mask.kind != "mask":
        raise ValueError("Inpainting requires an image and a selection mask, not a cutout.")
    if mask.purpose != "selection-mask" or mask.channel != "L":
        raise ValueError("Use an explicit grayscale selection mask, not a cutout or image alpha.")
    if (
        mask.source_image_id != source.artifact_id
        or mask.source_frame_id != source.frame_id
        or (mask.frame.width, mask.frame.height) != (source.frame.width, source.frame.height)
        or mask.frame.orientation != source.frame.orientation
    ):
        raise ValueError("The mask must match this exact oriented source image and frame.")
    return source, mask


def capture_mask_handoff(
    store,
    session_id,
    revision,
    source_id,
    mask_id,
    *,
    owner,
    choice,
    instance_count=1,
    instance_index=None,
    invert=False,
):
    """No resize, alpha extraction, preparation or compute occurs at this boundary."""
    if choice not in ("union", "instance"):
        raise ValueError("Choose the union mask or one explicit instance mask.")
    if type(instance_count) is not int or not 1 <= instance_count <= 64:
        raise ValueError("The mask result count must be from one through 64.")
    if choice == "union" and instance_index is not None:
        raise ValueError("A union mask has no instance index.")
    if choice == "instance" and (
        type(instance_index) is not int or not 0 <= instance_index < instance_count
    ):
        raise ValueError("Select one instance index from the captured mask results.")
    if type(invert) is not bool:
        raise ValueError("Mask inversion must be true or false.")
    source, mask = _refs(store, session_id, revision, source_id, mask_id, owner)
    if type(mask.inverted) is not bool:
        raise ValueError("The stored selection mask has no valid inversion convention.")
    return ResultMaskHandoff(
        session_id,
        revision,
        source_id,
        mask_id,
        source.frame_id,
        mask.frame_id,
        choice,
        instance_index,
        instance_count,
        mask.inverted ^ invert,
    )


def inpaint_inputs(store, handoff, *, owner):
    """Map white selection to the editor's layer alpha, exactly once."""
    from PIL import Image, ImageOps

    if not isinstance(handoff, ResultMaskHandoff):
        raise ValueError("Capture an owned result-mask handoff before inpainting.")
    source, mask = _refs(
        store,
        handoff.session_id,
        handoff.revision,
        handoff.source_id,
        handoff.mask_id,
        owner,
    )
    if (source.frame_id, mask.frame_id) != (handoff.source_frame_id, handoff.mask_frame_id):
        raise ValueError("The captured image or selection mask frame changed.")
    if source.frame.dtype != "uint8" or source.frame.color_space != "sRGB":
        raise ValueError("Use an explicit 8-bit sRGB working image for this mask handoff.")
    with Image.open(source.path) as value:
        image = ImageOps.exif_transpose(value).convert("RGBA").copy()
    with Image.open(mask.path) as value:
        if value.mode not in ("1", "L"):
            raise ValueError("The selection mask must remain a single grayscale channel.")
        selection = ImageOps.exif_transpose(value).convert("L").copy()
    if image.size != selection.size:
        raise ValueError("The selection mask must have the exact source dimensions.")
    if handoff.inverted:
        selection = ImageOps.invert(selection)
    if selection.getbbox() is None:
        raise ValueError("The selection mask is empty; no inpaint operation was submitted.")
    # Recheck signatures after reading; a changed file never crosses into preparation.
    _refs(
        store,
        handoff.session_id,
        handoff.revision,
        handoff.source_id,
        handoff.mask_id,
        owner,
    )
    layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
    layer.putalpha(selection)
    return dict(
        main_view=image,
        inpaint_view={"layers": [layer]},
        inpaint_toggle=True,
        input_image=None,
    )

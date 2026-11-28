"""Explicit API media enters the same frozen guide request as owned UI results."""


def capture(data, payload, *, decode_image, manager, family, owner):
    options = payload.get("control_guide")
    if options is None:
        return data
    if (
        not isinstance(options, dict)
        or set(options) - {"image", "kind", "convention", "denoise"}
        or not {"image", "kind", "convention"} <= set(options)
    ):
        raise ValueError("Provide only explicit guide image, kind, convention and edit denoise.")
    if payload.get("editor_context") is not None or not payload.get("input_image"):
        raise ValueError(
            "Uploaded guide editing requires an explicit original without Editor handles."
        )
    if data.get("image_number", 1) != 1 or payload.get("mask") or payload.get("borders"):
        raise ValueError("Uploaded guide editing returns one image without an inpaint mask.")
    controls = dict(data.get("controlnet") or {})
    if controls.get("type") not in ("depth", "control") or not controls.get("model"):
        raise ValueError("Select an explicit installed depth/normal ControlNet for the guide.")
    if controls.get("depth_preprocess", "none") != "none":
        raise ValueError("A precomputed guide cannot run another depth estimator.")
    from ruinedfooocus.features.image_editing.control_candidates import selected_control
    from ruinedfooocus.features.image_editing.guide_handoffs import capture_uploaded_guide_inputs

    _, signature, _ = selected_control(manager, controls["model"], family)
    controls.update(model_signature=list(signature), depth_preprocess="none")
    guide = decode_image(
        options["image"], preserve_mode=True, require_8bit=True, max_pixels=16777216
    )
    result = capture_uploaded_guide_inputs(
        source=data.get("input_image"),
        guide=guide,
        kind=options["kind"],
        convention=options["convention"],
        denoise=options.get("denoise", 0.35),
        data=dict(data, controlnet=controls),
        manager=manager,
        family=family,
        owner=owner,
    )
    if payload.get("size") not in (None, "auto"):
        result["resolution_auto"] = False
    else:
        result.setdefault("resolution_auto", True)
    return result

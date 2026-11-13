from ruinedfooocus.app.state import state
import ruinedfooocus.app.state as shared

import ruinedfooocus.features.generation.image_pipeline as image_pipeline
import ruinedfooocus.features.automation.template_pipeline as template_pipeline
import ruinedfooocus.features.image_processing.upscale_pipeline as upscale_pipeline
import ruinedfooocus.features.automation.search_pipeline as search_pipeline
import ruinedfooocus.features.image_processing.rembg_pipeline as rembg_pipeline
import ruinedfooocus.features.image_processing.native_vision_pipeline as native_vision_pipeline
import ruinedfooocus.features.chat.llama_pipeline as llama_pipeline
import ruinedfooocus.features.automation.hashbang_pipeline as hashbang_pipeline
import ruinedfooocus.features.image_editing.controlnet as controlnet


class NoPipeLine:
    pipeline_type = []


def selected_pipeline():
    from ruinedfooocus.app.resource_lifecycle import remember_pipeline

    selected = state["pipeline"]
    remember_pipeline(selected)
    return selected


def update(gen_data):
    from ruinedfooocus.features.image_processing import image_composition
    from ruinedfooocus.features.image_processing import still_image_utilities as depth_utilities
    from ruinedfooocus.features.models.image_scope import (
        require_image_family,
        require_image_operation,
    )

    if gen_data.get("task_type") != "llama":
        require_image_operation(gen_data)

    prompt = gen_data["prompt"] if "prompt" in gen_data else ""
    if (
        "_api_job" in gen_data
        or gen_data.get("task_type") == "tool_call"
        or gen_data.get("_metadata_prompt")
    ):
        # External prompts and imported metadata are data, not local commands.
        prompt = ""
    cn_settings = controlnet.get_settings(gen_data)
    cn_type = cn_settings["type"] if "type" in cn_settings else ""
    composition_utility = image_composition.handles(cn_type)
    native_utility = cn_type.lower() in ("detect", "segment", "depth-map", "restore-seedvr2")
    depth_utility = depth_utilities.is_utility(cn_type)
    if depth_utility and "native_vision" in getattr(state.get("pipeline"), "pipeline_type", ()):
        from ruinedfooocus.features.image_processing.native_vision import clear_model
        from ruinedfooocus.features.image_processing.native_seedvr2 import (
            clear_model as clear_seedvr2,
        )

        clear_model(state["pipeline"])
        clear_seedvr2(state["pipeline"])
    if not depth_utility and "depth_utility" in getattr(state.get("pipeline"), "pipeline_type", ()):
        depth_utilities.clear_model(state["pipeline"])
    native_utility = native_utility or depth_utility
    if native_utility or composition_utility:
        prompt = ""  # Utilities never interpret prompt commands.
    if not native_utility and "native_vision" in getattr(
        state.get("pipeline"), "pipeline_type", ()
    ):
        from ruinedfooocus.features.image_processing.native_vision import clear_model

        clear_model(state["pipeline"])
        from ruinedfooocus.features.image_processing.native_seedvr2 import (
            clear_model as clear_seedvr2,
        )

        clear_seedvr2(state["pipeline"])
    if cn_type.lower() in ("upscale", "rembg"):
        prompt = ""  # An explicit utility cannot be redirected by a free prompt command.
    if (
        gen_data.get("task_type") != "llama"
        and gen_data.get("inpaint_toggle")
        and (native_utility or composition_utility or cn_type.lower() in ("upscale", "rembg"))
    ):
        raise ValueError("Turn off inpainting before running a standalone image tool.")

    try:
        if gen_data.get("task_type") != "llama" and cn_type.lower() == "faceswap":
            raise ValueError(
                "Face swapping is not available. Use Restore cropped face for restoration."
            )
        if "task_type" in gen_data and gen_data["task_type"] == "llama":
            if state["pipeline"] is None or "llama" not in state["pipeline"].pipeline_type:
                state["pipeline"] = llama_pipeline.pipeline()

        elif prompt.lower() == "ruinedfooocuslogo":
            if state["pipeline"] is None or "template" not in state["pipeline"].pipeline_type:
                state["pipeline"] = template_pipeline.pipeline()

        elif prompt.startswith("#!"):
            if state["pipeline"] is None or "hashbang" not in state["pipeline"].pipeline_type:
                state["pipeline"] = hashbang_pipeline.pipeline()

        elif prompt.lower().startswith("search:"):
            if state["pipeline"] is None or "search" not in state["pipeline"].pipeline_type:
                state["pipeline"] = search_pipeline.pipeline()

        elif cn_type.lower() == "upscale":
            if state["pipeline"] is None or "upscale" not in state["pipeline"].pipeline_type:
                state["pipeline"] = upscale_pipeline.pipeline()

        elif cn_type.lower() == "rembg":
            if state["pipeline"] is None or "rembg" not in state["pipeline"].pipeline_type:
                state["pipeline"] = rembg_pipeline.pipeline()

        elif composition_utility:
            from ruinedfooocus.features.image_processing import image_composition_pipeline

            if (
                state["pipeline"] is None
                or "image_composition" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = image_composition_pipeline.pipeline()

        elif native_utility:
            if depth_utility:
                from ruinedfooocus.features.image_processing import depth_utility_pipeline

                if (
                    state["pipeline"] is None
                    or "depth_utility" not in state["pipeline"].pipeline_type
                ):
                    state["pipeline"] = depth_utility_pipeline.pipeline()
            elif (
                state["pipeline"] is None or "native_vision" not in state["pipeline"].pipeline_type
            ):
                state["pipeline"] = native_vision_pipeline.pipeline()

        else:
            baseModel = None
            if "base_model_name" in gen_data:
                file = shared.models.get_file("checkpoints", gen_data["base_model_name"])
                if file is None:
                    file = ""
                    baseModel = "None"
                else:
                    path = shared.models.get_models_by_path("checkpoints", file)
                    baseModel = shared.models.get_model_base(path)
                baseModelName = gen_data["base_model_name"]
            require_image_family(baseModel)
            if state["pipeline"] is None:
                state["pipeline"] = NoPipeLine()
            if baseModel is not None:
                from ruinedfooocus.features.generation.sdcpp_pipeline import select_pipeline

                state["pipeline"] = select_pipeline(
                    gen_data,
                    state["pipeline"],
                    image_pipeline.pipeline,
                    values=getattr(getattr(shared, "settings", None), "default_settings", {}),
                )

        if state["pipeline"] is None or len(state["pipeline"].pipeline_type) == 0:
            print(f"Using default pipeline.")
            from ruinedfooocus.features.generation.sdcpp_pipeline import select_pipeline

            state["pipeline"] = select_pipeline(
                gen_data,
                state["pipeline"],
                image_pipeline.pipeline,
                values=getattr(getattr(shared, "settings", None), "default_settings", {}),
            )

        return selected_pipeline()
    except Exception as error:
        # Let the worker report the failure and restore controls for retry.
        raise RuntimeError(f"Could not select generation pipeline: {error}") from error

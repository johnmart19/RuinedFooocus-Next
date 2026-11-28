"""Compare pinned native registrations with RF routes using AST/data only.

Run explicitly after changing either native pin. This never imports application
or native modules, loads weights, probes devices, downloads, or enables a route.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

if __package__:
    from .native_capability_inventory import reconcile_inventory
else:
    from native_capability_inventory import reconcile_inventory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NATIVE_REPOSITORIES = {"native": "ComfyUI", "gguf": "molbal/ComfyUI-GGUF"}
CALLER_DIRECTORIES = (
    "backends/comfy",
    "features/generation",
    "features/image_editing",
    "features/image_processing",
    "features/models",
)


def assignments(nodes):
    found = {}
    for node in nodes:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and node.value is not None:
                found[target.id] = node.value
    return found


def expression(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def literal_sequence(node, constants):
    """Resolve literal discovery lists, including shared starred suffix tuples."""
    if isinstance(node, ast.Name):
        return literal_sequence(constants.get(node.id), constants)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result = []
        for item in node.elts:
            if isinstance(item, ast.Starred):
                result.extend(literal_sequence(item.value, constants))
            elif isinstance(item, ast.Constant) and isinstance(item.value, str):
                result.append(item.value)
            else:
                raise ValueError("Discovery suffix declaration needs an updated static audit")
        return result
    raise ValueError("Discovery suffix declaration is no longer a literal sequence")


def inherited(classes, name, attribute, method=False):
    node = classes.get(name)
    if node is None:
        return None, None
    found = (
        next(
            (
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == attribute
            ),
            None,
        )
        if method
        else assignments(node.body).get(attribute)
    )
    if found is not None:
        return found, name
    for base in node.bases:
        found, owner = inherited(classes, ast.unparse(base).split(".")[-1], attribute, method)
        if found is not None:
            return found, owner
    return None, None


def class_definitions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def named_function(tree, name):
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ),
        None,
    )


def code_body(function):
    return [
        statement
        for statement in function.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def direct_return(function, label):
    statements = code_body(function) if function is not None else []
    if len(statements) != 1 or not isinstance(statements[0], ast.Return):
        raise ValueError(f"{label}: expected a static return; registration needs review")
    return statements[0].value


def builtin_node_paths(core):
    loader = named_function(core, "init_builtin_extra_nodes")
    constants = assignments(loader.body) if loader is not None else {}
    names = literal_sequence(constants.get("extras_files"), {})
    # A reassignment, append, or different loader loop requires a review of the
    # denominator; reading the initial list alone would silently miss nodes.
    uses = [
        node
        for node in ast.walk(loader)
        if isinstance(node, ast.Name) and node.id == "extras_files"
    ]
    if (
        not names
        or len(set(names)) != len(names)
        or len(uses) != 2
        or not any(isinstance(node, ast.For) and node.iter in uses for node in loader.body)
        or any(not re.fullmatch(r"nodes_[a-z0-9_]+\.py", name) for name in names)
    ):
        raise ValueError("Built-in node loader paths need review")
    return [f"comfy_extras/{name}" for name in names]


def named_class_list(node, classes, label):
    if not isinstance(node, ast.List) or any(
        not isinstance(item, ast.Name) or item.id not in classes for item in node.elts
    ):
        raise ValueError(f"{label}: expected a literal list of local node classes")
    return node.elts


def v1_registrations(tree, classes, mapping, label):
    """Recognize a literal mapping and the pinned hooks' class-attribute loop."""
    if not isinstance(mapping, ast.Dict):
        raise ValueError(f"{label}: V1 registration mapping needs review")
    rows = [
        (expression(key), value, value.lineno) for key, value in zip(mapping.keys, mapping.values)
    ]
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        writes = [
            item
            for item in ast.walk(statement)
            if isinstance(item, ast.Name) and item.id == "NODE_CLASS_MAPPINGS"
        ]
        if not writes or (
            len(writes) == 1
            and isinstance(statement, (ast.Assign, ast.AnnAssign))
            and statement.value is mapping
        ):
            continue
        if not (
            isinstance(statement, ast.For)
            and not statement.orelse
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.iter, ast.Name)
            and len(writes) == 1
        ):
            raise ValueError(f"{label}: dynamic V1 registration needs review")
        iterator = statement.target.id
        update = next(
            (
                item
                for item in statement.body
                if isinstance(item, ast.Assign)
                and any(
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "NODE_CLASS_MAPPINGS"
                    for target in item.targets
                )
            ),
            None,
        )
        if (
            update is None
            or len(update.targets) != 1
            or not isinstance(update.value, ast.Name)
            or update.value.id != iterator
            or ast.unparse(update.targets[0].slice) != f"{iterator}.NodeId"
        ):
            raise ValueError(f"{label}: V1 registration loop needs review")
        for node in named_class_list(assignments(tree.body).get(statement.iter.id), classes, label):
            identifier, _ = inherited(classes, node.id, "NodeId")
            rows.append((schema_string(identifier), node, node.lineno))
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(node, ast.Name)
        or node.id not in classes
        for key, node, _ in rows
    ):
        raise ValueError(f"{label}: V1 registration ID or local class needs review")
    # A literal identifier or inherited class constant is required; `expression`
    # falls back to display text for other consumers and is not enough here.
    if any(not isinstance(key, ast.Constant) for key in mapping.keys):
        raise ValueError(f"{label}: nonliteral V1 mapping key needs review")
    return rows


def schema_string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def schema_metadata(classes, name):
    method, owner = inherited(classes, name, "define_schema", True)
    if method is None:
        return {}
    body = code_body(method)
    if (
        not body
        or not isinstance(body[-1], ast.Return)
        or sum(isinstance(node, ast.Return) for node in ast.walk(method)) != 1
    ):
        return {}
    returned = body[-1].value
    parent_name = next(
        (ast.unparse(base) for base in classes[owner].bases if ast.unparse(base) in classes), None
    )
    parent = None

    def parent_metadata():
        nonlocal parent
        if parent is None:
            parent = schema_metadata(classes, parent_name) if parent_name else {}
        return parent

    fields = {}
    if isinstance(returned, ast.Call) and ast.unparse(returned.func) in {"io.Schema", "IO.Schema"}:
        for keyword in returned.keywords:
            key, value = keyword.arg, keyword.value
            if key not in {"node_id", "category"}:
                continue
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                if value.value.id == method.args.args[0].arg:
                    value, _ = inherited(classes, name, value.attr)
                elif (
                    ast.unparse(assignments(body).get(value.value.id, ast.Constant(None)))
                    == "super().define_schema()"
                ):
                    fields[key] = parent_metadata().get(value.attr)
                    continue
            fields[key] = schema_string(value)
    elif (
        isinstance(returned, ast.Name)
        and ast.unparse(assignments(body).get(returned.id, ast.Constant(None)))
        == "super().define_schema()"
    ):
        fields.update(parent_metadata())
        for statement in body[:-1]:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == returned.id
                    ):
                        fields[target.attr] = schema_string(statement.value)
    return fields


def module_registrations(tree, label):
    classes = class_definitions(tree)
    mapping = assignments(tree.body).get("NODE_CLASS_MAPPINGS")
    if mapping is not None and not (isinstance(mapping, ast.Constant) and mapping.value is None):
        return "v1", classes, v1_registrations(tree, classes, mapping, label)
    factory = direct_return(named_function(tree, "comfy_entrypoint"), label)
    if not (
        isinstance(factory, ast.Call)
        and isinstance(factory.func, ast.Name)
        and factory.func.id in classes
        and not factory.args
        and not factory.keywords
    ):
        raise ValueError(f"{label}: V3 extension factory needs review")
    pending = list(classes[factory.func.id].bases)
    while pending:
        base = ast.unparse(pending.pop())
        if base == "ComfyExtension":
            break
        if base in classes:
            pending.extend(classes[base].bases)
    else:
        raise ValueError(f"{label}: V3 factory has no auditable ComfyExtension base")
    method, _ = inherited(classes, factory.func.id, "get_node_list", True)
    registered = direct_return(method, label)
    rows = []
    for item in named_class_list(registered, classes, label):
        node_id = schema_metadata(classes, item.id).get("node_id")
        if not node_id:
            raise ValueError(f"{label}: V3 schema ID for {item.id} needs review")
        rows.append((node_id, item, item.lineno))
    return "v3", classes, rows


# Workflow boundaries are reviewed explicitly. A node may be shared by a working
# path and a missing mode; names/categories and direct imports cannot settle that.
PIPELINE = "features/generation/image_pipeline.py:pipeline."
EDIT = "features/image_editing/"
PROCESS = "features/image_processing/"
OPERATION_FAMILIES = (
    # operation, role, disposition, native anchors (IDs or file:symbol), RF anchors, boundary/next check
    (
        "text-to-image",
        "generation",
        "source-route-present",
        "CLIPTextEncode EmptyLatentImage KSampler VAEDecode",
        (PIPELINE + "process",),
        "Check exact family, components, recipe and real output separately.",
    ),
    (
        "GLIGEN ordered text-box layout",
        "conditioning",
        "source-route-present",
        "GLIGENLoader GLIGENTextBoxApply native/sd.py:load_gligen",
        (
            PIPELINE + "process",
            EDIT + "gligen_layout.py:capture_assets",
            EDIT + "gligen_layout.py:apply",
            EDIT + "ui_native_auxiliary.py:create_ui",
        ),
        "Fresh eight-aligned target canvas; exact fuser order/context and unprojected pooled width are validated. Native model/device/visual acceptance remains pending.",
    ),
    (
        "native hypernetwork attention",
        "model-patch",
        "source-route-present",
        "HypernetworkLoader native/comfy_extras/nodes_hypernetwork.py:load_hypernetwork_patch",
        (
            PIPELINE + "process",
            EDIT + "native_hypernetwork.py:capture_assets",
            EDIT + "native_hypernetwork.py:patch",
            EDIT + "ui_native_auxiliary.py:create_ui",
        ),
        "Uses the native weights-only helper and attn1/attn2 callback on a request clone; incompatible dimensions or activation fail before sampling. Exact artifact/device/visual acceptance remains pending.",
    ),
    (
        "variations and image-to-image",
        "editing",
        "source-route-present",
        "VAEEncode KSamplerAdvanced",
        (PIPELINE + "process",),
        "Verify denoise, seed, source dimensions and replay on a matching artifact.",
    ),
    (
        "MiniMax H3 Image generation and ordered edit",
        "generation",
        "source-route-present",
        "MiniMaxH3ImageConditioning ConditioningZeroOut native/supported_models.py:MiniMaxH3Image",
        (
            "features/generation/minimax_image.py:capture",
            "features/generation/minimax_image.py:process",
            PIPELINE + "process",
            "app/api_runtime.py:_image_operation",
        ),
        "The released PDD image graph has a distinct conditioning encoder, image VAE and B,24,1,H/16,W/16 latent. Ordered references, exact framing and replay use captured source identities; additional graph combinations, real weights, device memory and visual acceptance remain separate.",
    ),
    (
        "masked editing and canvas extension",
        "editing",
        "source-route-present",
        "VAEEncodeForInpaint InpaintModelConditioning ImagePadForOutpaint",
        (PIPELINE + "process", EDIT + "inpaint_utils.py:prepare_outpaint"),
        "SDXL and Flux inpaint paths have separate conditioning; validate mask, untouched RGB and outpaint geometry.",
    ),
    (
        "SD1/SD2 nine-channel inpaint",
        "editing",
        "source-route-present",
        "InpaintModelConditioning native/supported_models.py:SD15 native/supported_models.py:SD20 native/supported_models_base.py:BASE.inpaint_model",
        (PIPELINE + "process", EDIT + "native_sd_inpaint.py:native_sd_inpaint_conditioning"),
        "The loaded SD15/SD20 nine-channel gate and matching four-channel eightfold VAE use native source/mask concat conditioning. Real model/mask/output qualification remains separate.",
    ),
    (
        "instruct image edit",
        "conditioning",
        "source-route-present",
        "InstructPixToPixConditioning",
        (EDIT + "native_conditioning.py:instruct_conditioning",),
        "Qualify the source image and both conditioning branches for each compatible checkpoint.",
    ),
    (
        "image-prompt and vision reference adapters",
        "conditioning",
        "source-route-present",
        "CLIPVisionEncode StyleModelApply PhotoMakerEncode unCLIPConditioning ReferenceMultiLatent TextEncodeQwenImageEditPlus Flux2Scheduler",
        (
            EDIT + "reference_adapters.py:apply_reference",
            EDIT + "gligen_layout.py:apply",
            EDIT + "native_conditioning.py:unclip_conditioning",
            EDIT + "ordered_native_references.py:OrderedNativeReferences.conditioning",
            EDIT + "ordered_native_references.py:OrderedNativeReferences.attach",
        ),
        "Keep StyleAdapter/Redux/PhotoMaker/UnCLIP separate from native edit recipes. Ordered Qwen Edit2509/2511 and Flux2/Klein references have typed native contracts; model defaults, real outputs and device acceptance remain separate.",
    ),
    (
        "Qwen Image 2.1 text conditioning",
        "conditioning",
        "source-route-present",
        "TextEncodeQwenImage21",
        (EDIT + "qwen_image21_conditioning.py:text_to_image_conditioning",),
        "The paired text caller does not pass reference images.",
    ),
    (
        "Qwen Image 2.1 reference edit",
        "conditioning",
        "source-route-present",
        "TextEncodeQwenImage21",
        (
            EDIT + "qwen_image21_conditioning.py:QwenImage21References.conditioning",
            PIPELINE + "process",
            PIPELINE + "textencode_pair",
        ),
        "Use paired image slots and RGBA reference latents with the native first-reference target grid. Qualify actual model output and device memory separately.",
    ),
    (
        "Qwen Image Layered",
        "layered-image",
        "source-route-present",
        "EmptyQwenImageLayeredLatentImage ReferenceLatent LatentCut LatentCutToBatch VAEDecode",
        (
            PIPELINE + "process",
            "features/generation/layered.py:capture",
            "features/generation/layered.py:reference_conditioning",
            "features/generation/layered.py:layer_latent",
            "features/models/native_component_layout.py:current_layout",
        ),
        "Named current-source Layered route captures RGBA components, separates repetition and layer counts, and removes reconstruction slot zero before ordered decode. Qualify real weights, visual quality and GPU stability separately; Auto remains disabled.",
    ),
    (
        "Joy Image Edit",
        "conditioning",
        "source-route-present",
        "TextEncodeJoyImageEdit",
        ("features/generation/image_pipeline.py:joy_reference_conditioning", PIPELINE + "process"),
        "Use the same source in both native encodes and sample a separate empty latent. Qualify real output and offload separately.",
    ),
    (
        "LongCat Image Edit",
        "conditioning",
        "source-route-present",
        "native/comfy/text_encoders/longcat_image.py:LongCatImageTokenizer.tokenize_with_weights native/supported_models.py:LongCatImage",
        (
            EDIT + "ordered_native_references.py:OrderedNativeReferences.conditioning",
            PIPELINE + "process",
        ),
        "The paired native image encoder passes vision images and reference latents with index method. Auto requires the edit source variant; explicit compatible recipes remain artifact-unqualified. Real output acceptance remains pending.",
    ),
    (
        "Cascade text to image",
        "multi-stage",
        "source-route-present",
        "StableCascade_EmptyLatentImage StableCascade_StageB_Conditioning",
        ("features/generation/cascade.py:decode_stage_b",),
        "Qualify bundled Stage C and Stage B with the documented two-stage recipe.",
    ),
    (
        "Cascade image-to-image and paired stage grids",
        "multi-stage",
        "source-route-present",
        "StableCascade_StageC_VAEEncode StableCascade_StageB_Conditioning",
        (
            "features/generation/cascade.py:source_latents",
            "features/generation/cascade.py:decode_stage_b",
            PIPELINE + "process",
        ),
        "The native Stage C encoder owns source resampling and paired C/B grids. Qualify real source-image output, denoise behavior and memory separately.",
    ),
    (
        "Cascade super-resolution control and vision mixing",
        "multi-stage",
        "source-route-present",
        "StableCascade_SuperResolutionControlnet ControlNetApplyAdvanced CLIPVisionEncode unCLIPConditioning",
        (
            "features/generation/cascade_guidance.py:capture_assets",
            "features/generation/cascade_guidance.py:stage_read_leases",
            "features/generation/cascade_guidance.py:super_resolution_source",
            "features/generation/cascade_guidance.py:apply_guidance",
            "features/generation/ui_cascade_guidance.py:create_ui",
            PIPELINE + "process",
        ),
        "Explicit native twofold EffNet control and ordered weighted CLIP vision conditioning reach Stage C; paired B grids and exact final target restoration retain native sampling/codec owners. Actual weights, output quality, device memory and rendered UI remain separate acceptance gates.",
    ),
    (
        "SDXL standalone refiner",
        "refinement",
        "source-route-present",
        "CLIPTextEncodeSDXLRefiner native/supported_models.py:SDXLRefiner",
        (EDIT + "native_conditioning.py:refiner_text_conditioning",),
        "Standalone text conditioning does not establish a base/refiner handoff.",
    ),
    (
        "SDXL base/refiner handoff",
        "multi-stage",
        "source-route-present",
        "KSamplerAdvanced CLIPTextEncodeSDXLRefiner native/supported_models.py:SDXLRefiner",
        (
            "features/generation/sdxl_refiner.py:capture_assets",
            "features/generation/sdxl_refiner.py:refiner_scope",
            "features/generation/sdxl_refiner.py:sample_handoff",
            "features/generation/ui_sdxl_refiner.py:create_ui",
            PIPELINE + "process",
        ),
        "Captured native base/refiner models share a checked sigma schedule; the base leaves residual noise and the refiner continues without added noise or intermediate codec conversion. Qualify actual weights, output quality and device memory separately.",
    ),
    (
        "Hunyuan Image 2.1 refiner",
        "refinement",
        "source-route-present",
        "HunyuanRefinerLatent",
        (EDIT + "hunyuan_refiner.py:refiner_conditioning",),
        "Qualify the dedicated image codec and source-noise contract; video nodes are separate.",
    ),
    (
        "ControlNet and T2I",
        "control",
        "source-route-present",
        "ControlNetLoader ControlNetApplyAdvanced native/comfy/controlnet.py:load_controlnet_state_dict",
        (PIPELINE + "refresh_controlnet",),
        "Use the native reader and matching base family; selection is not proof for every control checkpoint.",
    ),
    (
        "ControlNet Union modes",
        "control",
        "source-route-present",
        "SetUnionControlNetType",
        (EDIT + "native_control_recipes.py:apply",),
        "Explicit native Union mode uses a request-local ControlNet copy; exact-artifact execution remains unqualified.",
    ),
    (
        "Qwen Image Fun patch",
        "model-patch",
        "source-route-present",
        "ModelPatchLoader QwenImageDiffsynthControlnet",
        (EDIT + "model_patches.py:apply_scope",),
        "Original Qwen DiffSynth has typed local assets and its own guide/mask; Qwen Image 2.1 Union stays separate.",
    ),
    (
        "Qwen Image 2.1 Fun ControlNet Union",
        "model-patch",
        "source-route-present",
        "ModelPatchLoader ZImageFunControlnet",
        (EDIT + "qwen_image21_union.py:apply_union",),
        "Preserve the qualified Qwen 2.1 path; exact patch class, guide order and RGBA codec remain required.",
    ),
    (
        "Z-Image Fun patch",
        "model-patch",
        "source-route-present",
        "ModelPatchLoader ZImageFunControlnet",
        (EDIT + "model_patches.py:apply_scope",),
        "Z-Image Fun has separate guide/inpaint source, native mask polarity and full-call asset leases; real execution remains unqualified.",
    ),
    (
        "USO style reference patch",
        "model-patch",
        "source-route-present",
        "ModelPatchLoader USOStyleReference",
        (EDIT + "model_patches.py:apply_scope",),
        "USO selects its own typed SigLIP vision encoder and Flux style projector without invented strength controls.",
    ),
    (
        "Anima LLLite patch",
        "model-patch",
        "source-route-present",
        "ModelPatchLoader AnimaLLLiteApply",
        (EDIT + "model_patches.py:apply_scope",),
        "Anima LLLite captures named-key v2 local weights, its own guide/mask and sigma interval; image caches stay request-local.",
    ),
    (
        "Qwen AliMama inpaint and outpaint",
        "control",
        "source-route-present",
        "ControlNetInpaintingAliMamaApply VAEEncode SetLatentNoiseMask ImagePadForOutpaint",
        (
            EDIT + "native_control_recipes.py:source_latent",
            EDIT + "native_control_recipes.py:apply",
        ),
        "Qwen 68-input masked ControlNet owns its source and white-edit mask or explicit canvas edges; native RGB output, model/device acceptance remains pending.",
    ),
    (
        "native LoRA hook schedules",
        "hook",
        "source-route-present",
        "CreateHookLora CreateHookLoraModelOnly CreateHookModelAsLora CreateHookModelAsLoraModelOnly "
        "CreateHookKeyframe CreateHookKeyframesInterpolated CreateHookKeyframesFromFloats "
        "CombineHooks2 CombineHooks4 CombineHooks8 SetHookKeyframes SetClipHooks native/comfy_extras/nodes_hooks.py:SetModelHooksOnCond.attach_hook",
        (
            EDIT + "lora_hooks.py:capture_assets",
            EDIT + "lora_hooks.py:build_hooks",
            EDIT + "lora_hooks.py:request_scope",
            EDIT + "clip_hook_conditioning.py:direct_conditioning",
            EDIT + "reference_adapters.py:apply_reference",
            EDIT + "ui_lora_hooks.py:create_ui",
        ),
        "At most eight local LoRA/checkpoint sources use native keyframes on the primary model/external CLIP, including PhotoMaker and GLIGEN intervals. Packed GGUF hook updates lack a native adapter; real-model acceptance remains separate.",
    ),
    (
        "RT-DETR detection",
        "detection",
        "source-route-present",
        "RTDETR_detect DrawBBoxes",
        (PROCESS + "native_vision.py:process",),
        "The current route draws boxes and reports normalized detections.",
    ),
    (
        "SAM3 point segmentation",
        "segmentation",
        "source-route-present",
        "SAM3_Detect",
        (PROCESS + "native_vision.py:process",),
        "The current caller passes positive/negative points and requests a single combined mask.",
    ),
    (
        "SAM3 text, box and per-instance masks",
        "segmentation",
        "source-route-present",
        "CLIPTextEncode SAM3_Detect",
        (
            PROCESS + "sam_prompts.py:capture",
            PROCESS + "sam_prompts.py:detect",
            PROCESS + "sam_prompts.py:render",
            PROCESS + "native_vision.py:process",
            PROCESS + "ui_native_vision.py:create_ui",
            "app/api_runtime.py:_native_vision_operation",
        ),
        "Captured native text, boxes and per-instance masks reach the standalone UI/API and ordered output bridge. The selected SAM source must include its paired text encoder; exact artifacts, rendered UI and visual quality require separate acceptance.",
    ),
    (
        "DA3 and Lotus depth preprocessing",
        "depth",
        "source-route-present",
        "LoadDA3Model DA3Inference DA3Render LotusConditioning",
        (EDIT + "depth_control.py:prepare_depth", EDIT + "lotus_depth.py:prepare_lotus"),
        "Verify model-specific map semantics and ControlNet geometry.",
    ),
    (
        "WeVisOne detection and matting",
        "detection-and-matting",
        "source-route-present",
        "WeVisOneDetect native/sd.py:load_clip",
        (
            PROCESS + "wevisone.py:capture_assets",
            PROCESS + "wevisone.py:process",
            PROCESS + "ui_wevisone.py:create_ui",
            "app/api_runtime.py:_native_vision_operation",
        ),
        "Positive native 2B/4B encoder identity selects a standalone image utility, never a diffusion encoder slot. Frozen RGBA8 working inputs, bounded native masks and owned API outputs have source/CPU callers; exact model answers, rendered UI and device qualification remain separate.",
    ),
    (
        "standalone depth map export",
        "depth",
        "source-route-present",
        "DA3Inference DA3Render native/sd.py:load_diffusion_model",
        (
            EDIT + "depth_control.py:render_da3",
            PROCESS + "native_vision.py:process",
            PROCESS + "ui_native_vision.py:create_ui",
            "app/api_runtime.py:_native_vision_operation",
            "integrations/openai_api.py:DepthMapRequest",
        ),
        "Verify the original-size DA3 grayscale PNG in UI and API with installed weights. Standalone Lotus remains separate.",
    ),
    (
        "standalone Lotus depth",
        "depth",
        "source-route-present",
        "LotusConditioning BasicScheduler SetFirstSigma SamplerCustomAdvanced VAEEncode VAEDecode ImageInvert",
        (
            PROCESS + "utility_lotus.py:render",
            PROCESS + "depth_utilities.py:capture_assets",
            PROCESS + "ui_depth_utilities.py:create_ui",
            "app/api_runtime.py:_native_vision_operation",
            "integrations/openai_api.py:DepthUtilityRequest",
        ),
        "Verify native/app pixels with the exact Lotus depth checkpoint and SD VAE, including odd dimensions and alpha. CPU/source contracts do not qualify weights or hardware.",
    ),
    (
        "MoGe still-image depth",
        "depth",
        "source-route-present",
        "LoadMoGeModel MoGeInference MoGeRender",
        (
            PROCESS + "utility_moge.py:render",
            PROCESS + "depth_utilities.py:capture_assets",
            PROCESS + "ui_depth_utilities.py:create_ui",
            "integrations/openai_api.py:DepthUtilityRequest",
        ),
        "Qualify native normalized still depth for exact MoGe weights. Panorama and mesh export remain separate, unimplemented scopes.",
    ),
    (
        "MediaPipe face landmark overlay",
        "detection",
        "source-route-present",
        "LoadMediaPipeFaceLandmarker MediaPipeFaceLandmarker MediaPipeFaceMeshVisualize",
        (
            PROCESS + "utility_face.py:render",
            PROCESS + "depth_utilities.py:capture_assets",
            PROCESS + "ui_depth_utilities.py:create_ui",
            "integrations/openai_api.py:DepthUtilityRequest",
        ),
        "Qualify exact face bundles and empty/multiple-face images. This caller produces IMAGE overlays, not MESH or animation outputs.",
    ),
    (
        "SDPose still whole-body pose map",
        "pose",
        "source-route-present",
        "SDPoseKeypointExtractor SDPoseDrawKeypoints",
        (
            PROCESS + "utility_pose.py:render",
            PROCESS + "depth_utilities.py:capture_assets",
            PROCESS + "ui_depth_utilities.py:create_ui",
            "integrations/openai_api.py:DepthUtilityRequest",
        ),
        "Qualify the whole-body SDPose checkpoint with its bundled SD VAE. This route estimates one person from a full still image; detector-assisted multi-person and video remain separate.",
    ),
    (
        "Marigold V2 still albedo",
        "intrinsic-image",
        "source-route-present",
        "MarigoldV2PostProcess ConditioningLoader ModelSamplingAuraFlow ManualSigmas SamplerCustomAdvanced",
        (
            PROCESS + "utility_marigold.py:render",
            PROCESS + "utility_marigold.py:verify_component",
            PROCESS + "depth_utilities.py:capture_assets",
            PROCESS + "ui_depth_utilities.py:create_ui",
            "integrations/openai_api.py:DepthUtilityRequest",
        ),
        "Qualify the exact albedo LoRA, VAE and saved conditioning with a compatible Qwen Image Edit encoding. Keep depth/normals tasks and real-device acceptance separate.",
    ),
    (
        "RGB curves through native LUT shader consumption",
        "image-processing",
        "source-route-present",
        "CurveEditor GLSLShader",
        (
            PROCESS + "image_shaders.py:process",
            PROCESS + "ui_image_shaders.py:create_ui",
        ),
        "Four bounded native curve LUTs drive CPU texel-center sampling or the static GLSL recipe. CPU policy/fallback is authoritative; actual GL/platform pixel equivalence remains deferred.",
    ),
    (
        "named native GLSL image effects",
        "image-processing",
        "source-route-present",
        "GLSLShader ImageInvert ImageBlend",
        (
            PROCESS + "image_shaders.py:capture_assets",
            PROCESS + "image_effect_cpu.py:render",
            PROCESS + "ui_image_shaders.py:create_ui",
        ),
        "Only reviewed invert, BT.709 grayscale and posterize recipes, with matching finite CPU operators/fallback. One source/output, bounded shape, no arbitrary code. Actual EGL/GLES execution remains unqualified.",
    ),
    (
        "native still blur sharpen brightness and contrast",
        "image-processing",
        "source-route-present",
        "ImageBlur ImageSharpen AdjustBrightness AdjustContrast",
        (
            PROCESS + "image_filters.py:process",
            PROCESS + "ui_image_filters.py:create_ui",
            PROCESS + "still_image_utilities.py:capture_assets",
        ),
        "Source-sized still IMAGE operations with unchanged alpha and no model. Only the two pure dataset adjustment bodies are image-domain overrides; training remains separate.",
    ),
    (
        "source RGB and BT.709 histogram analysis",
        "image-analysis",
        "source-route-present",
        "ImageHistogram",
        (
            PROCESS + "image_histogram.py:analyze",
            PROCESS + "ui_image_histogram.py:create_ui",
            "integrations/openai_api.py:HistogramRequest",
        ),
        "Pure CPU 256-bin data output; encoded RGB8, transparent pixels included, source identity frozen. Curves need a separate pixel consumer.",
    ),
    (
        "BiRefNet background removal",
        "background-removal",
        "source-route-present",
        "LoadBackgroundRemovalModel RemoveBackground",
        (PROCESS + "native_background.py:remove_background",),
        "Qualify actual alpha behavior for matching local weights; ONNX is a separate caller.",
    ),
    (
        "pixel upscale and PiD restoration",
        "upscale",
        "source-route-present",
        "UpscaleModelLoader ImageUpscaleWithModel PiDConditioning",
        (
            PROCESS + "upscale_pipeline.py:pipeline.process",
            EDIT + "pid_conditioning.py:pid_conditioning",
        ),
        "Spandrel model and PiD diffusion routes have different required weights and dimensions.",
    ),
    (
        "SeedVR2 still-image restoration",
        "restoration",
        "source-route-present",
        "SeedVR2Preprocess SeedVR2Conditioning SeedVR2PostProcessing native/supported_models.py:SeedVR2",
        (
            PROCESS + "native_seedvr2.py:capture_assets",
            PROCESS + "native_seedvr2.py:process",
            PROCESS + "ui_native_seedvr2.py:create_ui",
            "app/api_runtime.py:_native_vision_operation",
        ),
        "Connected Image tools and API still route holds both captured files through native one-frame encode, conditioning, sampling and decode; real artifact and visual qualification remain separate. Temporal nodes remain excluded.",
    ),
    (
        "captioning and native text generation",
        "captioning",
        "specialized-consumer",
        "TextGenerateV2",
        (PROCESS + "interrogate.py:queued_caption",),
        "RF has its own captioners; native Comfy text generation is a separate candidate, not their implementation.",
    ),
    (
        "batching and advanced sampling",
        "sampling",
        "specialized-consumer",
        "KSamplerAdvanced BasicScheduler SamplerCustomAdvanced",
        (PIPELINE + "process", "features/generation/image_sampling.py:sample_omnigen2"),
        "RF exposes selected batch, schedule and guidance inputs; audit additional primitives by workflow.",
    ),
    (
        "checkpoint and diffusion-only loaders",
        "model-artifact",
        "source-route-present",
        "CheckpointLoaderSimple UNETLoader native/sd.py:load_checkpoint_guess_config native/sd.py:load_diffusion_model",
        (PIPELINE + "load_base_model",),
        "Bundled components take precedence; split weights need role-compatible companions.",
    ),
    (
        "local Diffusers bundle",
        "model-artifact",
        "source-route-present",
        "DiffusersLoader native/comfy/diffusers_load.py:load_diffusers",
        ("features/models/local_diffusers.py:discover_bundles", PIPELINE + "load_base_model"),
        "RF reads a bounded local layout through its own adapter; no general indexed/sharded or custom code support is implied.",
    ),
    (
        "text encoders and vision towers",
        "model-artifact",
        "source-route-present",
        "CLIPLoader DualCLIPLoader CLIPVisionLoader native/sd.py:load_clip",
        (PIPELINE + "load_base_model", EDIT + "reference_adapters.py:apply_reference"),
        "Text and vision tower roles, projector needs and encoder architectures are separate; suffix is only discovery evidence.",
    ),
    (
        "image VAE",
        "model-artifact",
        "source-route-present",
        "VAELoader native/sd.py:VAE",
        (PIPELINE + "load_base_model",),
        "Require matching latent channels, spatial layout and output codec; video is separate.",
    ),
    (
        "LoRA state dictionaries and embeddings",
        "model-artifact",
        "source-route-present",
        "LoraLoader native/sd.py:load_lora_for_models native/comfy/sd1_clip.py:load_embed",
        (PIPELINE + "load_loras", "backends/comfy/embedding_paths.py:register_embedding_paths"),
        "Compatibility depends on patch target or embedding dimensions, not a shared suffix.",
    ),
    (
        "GGUF diffusion and text encoders",
        "model-artifact",
        "source-route-present",
        "UnetLoaderGGUF DualCLIPLoaderGGUF gguf/loader.py:gguf_sd_loader gguf/loader.py:gguf_clip_loader",
        ("backends/comfy/gguf_loader.py:load_diffusion_model", PIPELINE + "load_base_model"),
        "Qualify encoder count, projector, quantization and device mode independently; other GGUF loader variants do not inherit this route.",
    ),
    (
        "GGUF VAE",
        "model-artifact",
        "needs-review",
        "VAELoaderGGUF gguf/nodes.py:VAELoaderGGUF.load_vae gguf/loader.py:gguf_sd_loader",
        (PIPELINE + "load_base_model",),
        "The native node only accepts the MiniMax H3 Q8_CR video VAE. RF's generic GGUF reader branch does not establish a still-image codec; review admission and casing.",
    ),
    (
        "GGUF LoRA",
        "model-artifact",
        "missing-app-route",
        "GGUFLoraImport gguf/lora.py:load_gguf_lora",
        (PIPELINE + "load_loras",),
        "The native import/target-mapping contract is separate from RF's state-dictionary path.",
    ),
    (
        "image-media open",
        "media-input",
        "specialized-consumer",
        "LoadImage LoadImageMask",
        (EDIT + "inpaint_utils.py:load_source",),
        "RF has its own reader; qualify browser eligibility, animation/frame selection, alpha, EXIF and color per format.",
    ),
    (
        "image-media export",
        "media-output",
        "specialized-consumer",
        "SaveImage",
        ("app/async_worker.py:_process",),
        "RF writes PNG through its own image worker. Audit export metadata and other formats separately from open eligibility and the native writer.",
    ),
)

NATIVE_ANCHOR_PATHS = {
    "native/comfy/controlnet.py": "comfy/controlnet.py",
    "native/comfy/diffusers_load.py": "comfy/diffusers_load.py",
    "native/comfy/sd1_clip.py": "comfy/sd1_clip.py",
    "native/comfy/text_encoders/longcat_image.py": "comfy/text_encoders/longcat_image.py",
    "native/folder_paths.py": "folder_paths.py",
    "gguf/__init__.py": "__init__.py",
    "gguf/nodes.py": "nodes.py",
    "gguf/loader.py": "loader.py",
    "gguf/lora.py": "lora.py",
}

EXCLUDED_MODULES = {
    "audio": {"ace", "audio", "audio_encoder", "lt_audio", "minimax_music", "yue2"},
    "video": {
        "ar_video",
        "camera_trajectory",
        "cosmos",
        "frame_interpolation",
        "lt",
        "lt_keyframes",
        "lt_upsampler",
        "minimax_h3",
        "mochi",
        "scail",
        "video",
        "video_model",
        "void",
        "wan",
        "wandancer",
        "wanmove",
        "webcam",
    },
    "3d": {
        "gaussian_splat",
        "hunyuan3d",
        "load_3d",
        "mesh_io",
        "mesh_postprocess",
        "sam3d_body",
        "save_3d",
        "stable3d",
        "trellis2",
        "triposplat",
    },
    "training": {"dataset", "train"},
}
DOMAIN_OVERRIDES = {
    "AdjustBrightness": "image",
    "AdjustContrast": "image",
    "DA3GeometryToMesh": "3d",
    "DA3GeometryToPointCloud": "3d",
    "LTXVLatentUpscaleModelLoaderGGUF": "video",
    "SAM3_VideoTrack": "video",
    "SAM3_TrackPreview": "video",
    "SAM3_TrackToMask": "video",
    "SeedVR2TemporalChunk": "video",
    "SeedVR2TemporalMerge": "video",
    "TextGenerateLTX2Prompt": "video",
    "VAELoaderGGUF": "video",
    "WanUni3CControlnetApply": "video",
}
ROLE_OVERRIDES = {
    "LoadImage": "media-input",
    "LoadImageMask": "media-input",
    "LoadImageOutput": "media-input",
    "SaveImage": "media-output",
    "PreviewImage": "preview",
    "ImagePadForOutpaint": "editing",
    "GGUFLoraImport": "lora-import",
    "TargetedQuantizationGGUF": "quantization-tool",
    "VAELoaderGGUF": "vae-loader",
    "LTXVLatentUpscaleModelLoaderGGUF": "latent-upscale-loader",
}
CATEGORY_ROLES = {
    "model/loaders": "model-loader",
    "model/latent": "latent",
    "model/conditioning": "conditioning",
    "model/sampling": "sampling",
    "model/patch": "model-patch",
    "model/merging": "model-merge",
    "model/training": "training",
    "advanced/hooks": "hook",
    "advanced/guidance": "guidance",
    "image/detection": "detection",
    "image/geometry estimation": "geometry-estimation",
    "image/background removal": "background-removal",
    "image/upscaling": "upscale",
    "image/mask": "mask",
    "image": "image-processing",
    "latent": "latent",
    "text": "text",
    "audio": "audio",
    "video": "video",
    "3d": "geometry",
    "util": "graph-utility",
    "utilities": "graph-utility",
}


def node_domain(node_id, path, category):
    if node_id in DOMAIN_OVERRIDES:
        return DOMAIN_OVERRIDES[node_id]
    if category == "bootleg":  # The pinned GGUF package uses one UI category for its tools.
        return "model"
    module = Path(path).stem.removeprefix("nodes_")
    for domain, modules in EXCLUDED_MODULES.items():
        if module in modules:
            return domain
    categories = category.casefold().split("/") if category else []
    if any("3d" in item or item == "stable zero123" for item in categories):
        return "3d"
    if any("audio" in item or "music" in item for item in categories):
        return "audio"
    if any("video" in item or item in {"ltxv", "wan", "mochi"} for item in categories):
        return "video"
    if categories and categories[0] == "image":
        return "image"
    if categories and categories[0] in {"model", "advanced", "latent", "loaders"}:
        return "model"
    if categories and categories[0] in {"util", "utilities"}:
        return "graph"
    if categories and categories[0] == "text":
        return "text"
    return "unclassified"


def node_role(node_id, category):
    if node_id in ROLE_OVERRIDES:
        return ROLE_OVERRIDES[node_id]
    if category == "bootleg" and node_id.startswith("UnetLoaderGGUF"):
        return "diffusion-loader"
    if category == "bootleg" and "CLIPLoaderGGUF" in node_id:
        return "text-encoder-loader"
    if category:
        for prefix, role in CATEGORY_ROLES.items():
            if category == prefix or category.startswith(prefix + "/"):
                return role
    return "unclassified"


def native_caller_references(trees):
    references = {}
    for path, tree in trees.items():
        if not path.startswith("rf-callers/"):
            continue
        used_names = {
            item.id
            for item in ast.walk(tree)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
        }
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom) and item.level == 0 and item.module:
                for alias in item.names:
                    if (alias.asname or alias.name) in used_names:
                        references.setdefault((item.module, alias.name), []).append(
                            {
                                "path": path.replace("rf-callers/", "ruinedfooocus/", 1),
                                "line": item.lineno,
                            }
                        )
    return references


def source_definition(trees, reference):
    path, _, symbol = reference.partition(":")
    tree = trees.get(path)
    for name in symbol.split("."):
        if tree is None:
            return None
        tree = next((item for item in tree.body if getattr(item, "name", None) == name), None)
    return tree


def source_anchor(trees, reference):
    path, _, symbol = reference.partition(":")
    node = source_definition(trees, reference)
    return {"source_key": path, "symbol": symbol, "line": node.lineno} if node else None


def calls_function(node, name):
    return node is not None and any(
        isinstance(item, ast.Call) and ast.unparse(item.func) == name for item in ast.walk(node)
    )


def calls_imported_function(node, module, name):
    return node is not None and any(
        calls_function(node, alias.asname or alias.name)
        for item in ast.walk(node)
        if isinstance(item, ast.ImportFrom) and item.level == 0 and item.module == module
        for alias in item.names
        if alias.name == name
    )


def gguf_lora_operation(trees, disposition, app, next_action):
    """Support older RF sources as well as the explicit native import caller."""
    pipeline = source_definition(trees, "rf-callers/" + PIPELINE + "load_loras")
    reference = "backends/comfy/gguf_lora.py:apply_gguf_lora"
    backend = source_definition(trees, "rf-callers/" + reference)
    if backend is None and not calls_function(pipeline, "apply_gguf_lora"):
        return disposition, app, next_action
    if calls_imported_function(
        pipeline, "ruinedfooocus.backends.comfy.gguf_lora", "apply_gguf_lora"
    ) and calls_imported_function(backend, "molbal_comfyui_gguf.lora", "load_gguf_lora"):
        return (
            "source-route-present",
            (*app, reference),
            "Qualify the captured file, native factor remapping and exact patch targets separately from ordinary LoRA and real-model output.",
        )
    return "needs-review", app, "Verify that the app dispatch reaches the pinned GGUF LoRA reader."


def gguf_vae_operation(trees, disposition, app, next_action):
    loader = source_definition(trees, "gguf/nodes.py:VAELoaderGGUF.load_vae")
    required_guards = {
        "extra['arch_str'] != 'minimax_h3_vae'",
        "extra.get('gguf_quant_mode') != 'int8_convrot'",
    }
    guarded = (
        {
            ast.unparse(item.test)
            for item in loader.body
            if isinstance(item, ast.If) and any(isinstance(step, ast.Raise) for step in item.body)
        }
        if loader is not None
        else set()
    )
    if not required_guards <= guarded:
        return (
            "needs-review",
            app,
            "Recheck the pinned GGUF VAE node's architecture and quantization requirements.",
        )
    pipeline = source_definition(trees, "rf-callers/" + PIPELINE + "load_base_model")
    if calls_imported_function(
        pipeline, "ruinedfooocus.core.model_formats", "require_image_vae_format"
    ):
        return (
            "unsupported",
            (*app, "features/models/image_components.py:inventory"),
            "Image slots require a compatible native state-dictionary VAE. The pinned GGUF node serves the MiniMax H3 Q8_CR video codec.",
        )
    return disposition, app, next_action


def operation_rows(trees, nodes):
    rows = []
    for name, role, disposition, native, app, next_action in OPERATION_FAMILIES:
        if name == "GGUF LoRA":
            disposition, app, next_action = gguf_lora_operation(
                trees, disposition, app, next_action
            )
        elif name == "GGUF VAE":
            disposition, app, next_action = gguf_vae_operation(trees, disposition, app, next_action)
        native_references = []
        app_references = []
        unresolved = []
        for ref in native.split():
            anchor = source_anchor(trees, ref) if ":" in ref else nodes.get(ref)
            if anchor:
                native_references.append(
                    anchor if ":" in ref else {"node_id": ref, **anchor["source"]}
                )
            else:
                unresolved.append(ref)
        for ref in app:
            anchor = source_anchor(trees, "rf-callers/" + ref)
            if anchor:
                app_references.append(anchor)
            else:
                unresolved.append(ref)
        rows.append(
            {
                "operation": name,
                "role": role,
                "disposition": "needs-review" if unresolved else disposition,
                "evidence_level": "source-only",
                "native_sources": native_references,
                "RF_sources": app_references,
                "unresolved_anchors": unresolved,
                "next_check": next_action,
            }
        )
    return rows


def folder_role_declarations(trees):
    """Folder buckets list candidates, never reader or architecture compatibility."""
    source = "native/folder_paths.py"
    tree = trees.get(source)
    rows = []
    if tree:
        constants = assignments(tree.body)
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if (
                not isinstance(target, ast.Subscript)
                or ast.unparse(target.value) != "folder_names_and_paths"
            ):
                continue
            role = schema_string(target.slice)
            if (
                not role
                or not isinstance(statement.value, ast.Tuple)
                or len(statement.value.elts) != 2
            ):
                raise ValueError("Native model folder declaration needs review")
            _, candidates = statement.value.elts
            rows.append(
                {
                    "role": role,
                    "native_discovery_tokens": []
                    if ast.unparse(candidates) == "set()"
                    else sorted(literal_sequence(candidates, constants)),
                    "source": {"source_key": source, "line": statement.lineno},
                }
            )
    for statement in trees["gguf/nodes.py"].body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if (
            not isinstance(call, ast.Call)
            or ast.unparse(call.func) != "update_folder_names_and_paths"
        ):
            continue
        role = schema_string(call.args[0]) if len(call.args) == 2 else None
        if not role or call.keywords:
            raise ValueError("GGUF model folder declaration needs review")
        rows.append(
            {
                "role": role,
                "native_base_roles": literal_sequence(call.args[1], {}),
                "source": {"source_key": "gguf/nodes.py", "line": statement.lineno},
            }
        )
    return rows


def derive_node_inventory(sources):
    """Derive the finite core, explicit bundled extras, and pinned GGUF registries."""
    trees = {name: ast.parse(source, filename=name) for name, source in sources.items()}
    core = trees["native/nodes.py"]
    extras = [f"native/{path}" for path in builtin_node_paths(core)]
    package = trees["gguf/__init__.py"]
    if not any(
        isinstance(item, ast.ImportFrom)
        and item.level == 1
        and item.module == "nodes"
        and any(
            alias.name == "NODE_CLASS_MAPPINGS" and alias.asname is None for alias in item.names
        )
        for item in ast.walk(package)
    ):
        raise ValueError("GGUF package node export needs review")
    callers = native_caller_references(trees)
    nodes = {}
    modules = []
    for source in ["native/nodes.py", *extras, "gguf/nodes.py"]:
        prefix, path = source.split("/", 1)
        scope = "gguf" if prefix == "gguf" else "core" if path == "nodes.py" else "bundled"
        mode, classes, registered = module_registrations(trees[source], source)
        modules.append(
            {
                "source_key": source,
                "scope": scope,
                "registration": mode,
                "node_count": len(registered),
            }
        )
        module_name = ("molbal_comfyui_gguf." if scope == "gguf" else "") + path.removesuffix(
            ".py"
        ).replace("/", ".")
        for node_id, item, line in registered:
            if node_id in nodes:
                raise ValueError(
                    f"{source}: duplicate node ID {node_id!r}; registration order needs review"
                )
            category_node, _ = inherited(classes, item.id, "CATEGORY")
            category = schema_string(category_node) or schema_metadata(classes, item.id).get(
                "category"
            )
            references = sorted(
                callers.get((module_name, item.id), ()), key=lambda ref: (ref["path"], ref["line"])
            )
            domain, role = node_domain(node_id, path, category), node_role(node_id, category)
            if domain in EXCLUDED_MODULES:
                disposition = "excluded-from-image-audit"
                reason = f"Native {domain} operation is outside the still-image audit scope."
            elif references:
                disposition = "specialized-consumer"
                reason = "A scoped RF module imports and references this class; inspect its call and supported modes."
            elif role in {"graph-utility", "hook", "latent", "sampling"}:
                disposition = "internal-primitive"
                reason = "Graph building block; review as part of a named workflow before exposing an entry point."
            else:
                disposition = "needs-review"
                reason = "No qualifying RF node caller found in the scoped sources; check named operations and alternate APIs."
            nodes[node_id] = {
                "id": node_id,
                "class": item.id,
                "source": {"source_key": source, "line": classes[item.id].lineno},
                "registration_line": line,
                "scope": scope,
                "registration": mode,
                "native_category": category,
                "native_domain": domain,
                "role": role,
                "disposition": disposition,
                "reason": reason,
                "RF_source_references": references,
            }
    operations = operation_rows(trees, nodes)
    for row in operations:
        for native_source in row["native_sources"]:
            if "node_id" in native_source:
                nodes[native_source["node_id"]].setdefault("operation_families", []).append(
                    row["operation"]
                )
    return {
        "node_inventory_schema": 1,
        "native_node_modules": modules,
        "native_nodes": sorted(nodes.values(), key=lambda node: (node["scope"], node["id"])),
        "native_folder_roles": folder_role_declarations(trees),
        "operation_families": operations,
        "inventory_scope": "Source declarations only. Hosted API nodes, custom nodes other than the pinned GGUF adapter, and workflow templates are outside this registry denominator. No import or device eligibility is implied.",
    }


def function_summary(tree, name):
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == name
        ),
        None,
    )
    if node is None:
        return None
    return {
        "line": node.lineno,
        "explicit_suffix_branches": sorted(
            {
                item.value
                for item in ast.walk(node)
                if isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and re.fullmatch(r"\.[a-z0-9]{2,12}", item.value)
            }
        ),
        "calls": sorted(
            {ast.unparse(item.func) for item in ast.walk(node) if isinstance(item, ast.Call)}
        ),
    }


def derive_sources(sources):
    """Pure comparison interface, also usable with finite source fixtures."""
    trees = {name: ast.parse(text) for name, text in sources.items()}
    native = trees["native/supported_models.py"]
    classes = {node.name: node for node in native.body if isinstance(node, ast.ClassDef)}
    classes.update(
        {
            node.name: node
            for node in trees["native/supported_models_base.py"].body
            if isinstance(node, ast.ClassDef)
        }
    )
    ordered = [node.id for node in assignments(native.body)["models"].elts]
    image = trees["rf/image_components.py"]
    image_data = assignments(image.body)
    components = json.loads(json.dumps(expression(image_data["_COMPONENT_KEYS"])))
    aliases = expression(image_data["ALIASES"])
    pipeline = next(
        node
        for node in trees["rf/image_pipeline.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "pipeline"
    )
    routes = assignments(pipeline.body)["known_model_info"]
    route_names = [expression(key) for key in routes.keys]
    used_clip_types = sorted(
        {
            node.attr
            for node in ast.walk(routes)
            if isinstance(node, ast.Attribute) and ast.unparse(node.value).endswith("CLIPType")
        }
    )
    native_clip = next(
        node
        for node in trees["native/sd.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "CLIPType"
    )
    clip_types = {name: expression(node) for name, node in assignments(native_clip.body).items()}
    detector_aliases = {}
    for node in ast.walk(trees["rf/model_architecture.py"]):
        if isinstance(node, ast.Dict):
            candidate = expression(node)
            if isinstance(candidate, dict) and "SD15" in candidate:
                detector_aliases = candidate
    records = []
    for name in ordered:
        node = classes[name]
        factory, owner = inherited(classes, name, "get_model", True)
        clip, clip_owner = inherited(classes, name, "clip_target", True)
        match, _ = inherited(classes, name, "unet_config")
        factories = (
            sorted(
                {
                    ast.unparse(item.func).split(".")[-1]
                    for item in ast.walk(factory)
                    if isinstance(item, ast.Call)
                    and ast.unparse(item.func).startswith("model_base.")
                }
            )
            if factory
            else []
        )
        if owner == "BASE":
            noise, _ = inherited(classes, name, "noise_aug_config")
            factories = ["BaseModel" if expression(noise) is None else "SD21UNCLIP"]
        label = detector_aliases.get(name, name)
        canonical = aliases.get(label, label)
        admitted = sorted(set(factories) & set(route_names))
        contract = {
            "match": expression(match),
            "factories": factories,
            "factory": ast.dump(factory, include_attributes=False) if factory else None,
            "clip": ast.dump(clip, include_attributes=False) if clip else None,
        }
        records.append(
            {
                "class": name,
                "line": node.lineno,
                "parents": [ast.unparse(base) for base in node.bases],
                "unet_match": expression(match),
                "runtime_factories": factories,
                "factory_owner": owner,
                "factory_line": factory.lineno if factory else None,
                "clip_target_owner": clip_owner,
                "clip_target_line": clip.lineno if clip else None,
                "contract_sha256": hashlib.sha256(
                    json.dumps(contract, sort_keys=True).encode()
                ).hexdigest(),
                "RF_runtime_classes": admitted,
                "RF_detector_label": label,
                "RF_component_canonical": canonical,
                "RF_component_roles": components.get(canonical),
                "route_status": "runtime-class-and-component-route-present"
                if admitted and canonical in components
                else "component-normalization-needs-review"
                if admitted
                else "native-task-route-needs-review",
            }
        )
    constants = (
        assignments(trees["rf/model_formats.py"].body) if "rf/model_formats.py" in trees else {}
    )
    formats = sorted(
        {
            suffix
            for item in ast.walk(trees["rf/model_handler.py"])
            if isinstance(item, ast.Assign)
            for target in item.targets
            if isinstance(target, ast.Attribute) and target.attr == "EXTENSIONS"
            for suffix in literal_sequence(item.value, constants)
        }
    )
    return {
        "native_config_classes": records,
        "native_clip_types": clip_types,
        "RF_runtime_configurations": route_names,
        "RF_component_configurations": components,
        "RF_aliases": aliases,
        "RF_image_clip_types": used_clip_types,
        "loader_contract": {
            "RF_discovery_extensions": formats,
            "native_weight_reader": function_summary(trees["native/utils.py"], "load_torch_file"),
            "native_checkpoint": function_summary(
                trees["native/sd.py"], "load_checkpoint_guess_config"
            ),
            "native_diffusion": function_summary(trees["native/sd.py"], "load_diffusion_model"),
            "native_text_encoder": function_summary(trees["native/sd.py"], "load_clip"),
            "RF_checkpoint_admission": function_summary(
                trees["rf/image_pipeline.py"], "load_base_model"
            ),
            "note": "Suffix discovery and explicit reader branches are not universal format/model/device compatibility. Inspect quantization/state-dict and operation contracts.",
        },
    }


def compare_baseline(previous, current):
    before = {item["class"]: item for item in previous["native_config_classes"]}
    after = {item["class"]: item for item in current["native_config_classes"]}
    added, removed = sorted(after.keys() - before.keys()), sorted(before.keys() - after.keys())
    changed = sorted(
        name
        for name in after.keys() & before.keys()
        if after[name]["contract_sha256"] != before[name]["contract_sha256"]
    )
    drift = {
        "added_config_classes": added,
        "removed_config_classes": removed,
        "changed_native_contracts": changed,
        "possible_renames_require_review": [
            [old, new]
            for old in removed
            for new in added
            if before[old]["contract_sha256"] == after[new]["contract_sha256"]
        ],
        "added_CLIP_types": sorted(
            current["native_clip_types"].keys() - previous["native_clip_types"].keys()
        ),
        "removed_CLIP_types": sorted(
            previous["native_clip_types"].keys() - current["native_clip_types"].keys()
        ),
        "changed_audited_native_sources": sorted(
            name
            for name in current.get("input_sha256", {})
            if name.startswith(("native/", "gguf/"))
            and name in previous.get("input_sha256", {})
            and current["input_sha256"][name] != previous["input_sha256"][name]
        ),
        "RF_added_runtime_routes": sorted(
            set(current["RF_runtime_configurations"]) - set(previous["RF_runtime_configurations"])
        ),
        "RF_removed_runtime_routes": sorted(
            set(previous["RF_runtime_configurations"]) - set(current["RF_runtime_configurations"])
        ),
        "RF_added_component_routes": sorted(
            current["RF_component_configurations"].keys()
            - previous["RF_component_configurations"].keys()
        ),
        "RF_removed_component_routes": sorted(
            previous["RF_component_configurations"].keys()
            - current["RF_component_configurations"].keys()
        ),
        "RF_changed_component_roles": sorted(
            name
            for name in current["RF_component_configurations"].keys()
            & previous["RF_component_configurations"].keys()
            if current["RF_component_configurations"][name]
            != previous["RF_component_configurations"][name]
        ),
        "policy": "Drift is a review queue. No removed/deprecated declaration, catalogue edit or route promotion occurs automatically.",
    }
    if "node_inventory_schema" in current:
        if previous.get("node_inventory_schema") != current["node_inventory_schema"]:
            drift["inventory_baseline_needs_review"] = [
                "Earlier report has no comparable node inventory; create a new inventory baseline."
            ]
        else:
            old_nodes = {row["id"]: row for row in previous["native_nodes"]}
            new_nodes = {row["id"]: row for row in current["native_nodes"]}
            fields = (
                "class",
                "scope",
                "registration",
                "native_category",
                "native_domain",
                "role",
                "disposition",
            )
            drift.update(
                added_native_node_ids=sorted(new_nodes.keys() - old_nodes.keys()),
                removed_native_node_ids=sorted(old_nodes.keys() - new_nodes.keys()),
                changed_native_node_rows=sorted(
                    key
                    for key in new_nodes.keys() & old_nodes.keys()
                    if new_nodes[key]["source"]["source_key"]
                    != old_nodes[key]["source"]["source_key"]
                    or any(new_nodes[key][field] != old_nodes[key][field] for field in fields)
                ),
            )
            old_operations = {
                row["operation"]: row["disposition"] for row in previous["operation_families"]
            }
            new_operations = {
                row["operation"]: row["disposition"] for row in current["operation_families"]
            }
            drift["changed_operation_dispositions"] = sorted(
                name
                for name in old_operations.keys() | new_operations.keys()
                if old_operations.get(name) != new_operations.get(name)
            )
    if "finite_inventory" in current:
        old_inventory = previous.get("finite_inventory")
        new_inventory = current["finite_inventory"]
        if (
            not old_inventory
            or old_inventory.get("schema_version") != new_inventory["schema_version"]
        ):
            drift["finite_inventory_baseline_needs_review"] = [
                "Create a comparable finite recipe/role baseline; earlier node counts do not establish it."
            ]
        else:
            for kind in ("blueprints", "official_templates"):
                old = {row["id"]: row for row in old_inventory[kind]["recipes"]}
                new = {row["id"]: row for row in new_inventory[kind]["recipes"]}
                drift[f"added_{kind}"] = sorted(new.keys() - old.keys())
                drift[f"removed_{kind}"] = sorted(old.keys() - new.keys())
                drift[f"changed_{kind}"] = sorted(
                    key for key in new.keys() & old.keys() if new[key] != old[key]
                )
    return drift


def checked_commit(root, name, pin):
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=20
    ).strip()
    if actual != pin:
        raise ValueError(
            f"{name} checkout {actual} differs from declared pin {pin}; no audit result is promoted"
        )
    return actual


def read_pin_sources(root, name, commit, paths):
    command = ["git", "-C", str(root)]
    entries = subprocess.check_output(
        [*command, "ls-tree", "-r", "-z", commit, "--", *paths], timeout=20
    ).split(b"\0")
    tracked = {}
    for entry in filter(None, entries):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        if kind == "blob" and mode in {"100644", "100755"}:
            tracked[path.decode("utf-8")] = oid
    if missing := sorted(set(paths) - tracked.keys()):
        raise ValueError(f"{name}: audited source absent from pin: {', '.join(missing)}")
    if subprocess.run(
        [*command, "diff", "--quiet", "--no-ext-diff", "--no-textconv", commit, "--", *paths],
        timeout=20,
    ).returncode:
        raise ValueError(
            f"{name}: audited native source differs from the pin; keep local modifications separate"
        )
    contents = {}
    for path in paths:
        source = root / path
        if source.is_symlink():
            raise ValueError(f"{name}: audited native source {path} must be a regular file")
        local = source.read_bytes()
        oid = tracked[path]
        algorithm = "sha1" if len(oid) == 40 else "sha256"
        if not any(
            hashlib.new(
                algorithm, f"blob {len(candidate)}\0".encode("ascii") + candidate
            ).hexdigest()
            == oid
            for candidate in (local, local.replace(b"\r\n", b"\n"))
        ):
            raise ValueError(f"{name}: audited native source {path} differs from its pinned blob")
        contents[path] = local
    return contents


def read_official_templates(templates_root=None):
    """Read installed data/RECORD identities without importing the template package."""
    if templates_root is None:
        packages = Path(sys.prefix) / "Lib/site-packages"
        if not packages.is_dir():
            candidates = sorted((Path(sys.prefix) / "lib").glob("python*/site-packages"))
            packages = next((path for path in candidates if path.is_dir()), packages)
        templates_root = packages / "comfyui_workflow_templates_json/templates"
    else:
        templates_root = templates_root.resolve()
        packages = templates_root.parent.parent
    manifest_path = packages / "comfyui_workflow_templates_core/manifest.json"
    if not templates_root.is_dir() or not manifest_path.is_file():
        return {}, {
            "disposition": "blocked",
            "reason": "Installed official template data/manifest unavailable; registry denominator remains separate.",
        }
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    declared = {}
    for template in manifest["templates"]:
        for asset in template.get("assets", []):
            filename = asset["filename"]
            if filename.endswith(".json"):
                if filename in declared and declared[filename] != asset["sha256"]:
                    raise ValueError(f"Official template {filename}: conflicting manifest digests")
                declared[filename] = asset["sha256"]
    documents, mismatches = {}, []
    for filename, digest in sorted(declared.items()):
        if Path(filename).name != filename:
            raise ValueError("Official template manifest filename must be a basename")
        path = templates_root / filename
        raw = path.read_bytes()
        observed_digest = hashlib.sha256(raw).hexdigest()
        if observed_digest != digest:
            mismatches.append(
                {
                    "filename": filename,
                    "manifest_sha256": digest,
                    "observed_sha256": observed_digest,
                }
            )
        documents[filename] = raw
    observed = {path.name for path in templates_root.glob("*.json")}
    if observed != declared.keys():
        raise ValueError("Official template JSON set differs from its installed manifest")
    versions = {}
    for name in (
        "comfyui_workflow_templates",
        "comfyui_workflow_templates_core",
        "comfyui_workflow_templates_json",
    ):
        for metadata in sorted(packages.glob(name + "-*.dist-info/METADATA")):
            for line in metadata.read_text(encoding="utf-8").splitlines():
                if line.startswith("Version: "):
                    versions[name] = line.removeprefix("Version: ")
                    break
    return documents, {
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "packages": versions,
        "manifest_json_assets": len(documents),
        "non_recipe_assets": [
            name
            for name, raw in documents.items()
            if not isinstance(json.loads(raw), dict) or "nodes" not in json.loads(raw)
        ],
        "all_json_assets_match_manifest": not mismatches,
        "manifest_mismatches": mismatches,
        "recipe_assets_match_manifest": not any(
            "nodes" in json.loads(documents[row["filename"]]) for row in mismatches
        ),
    }


def read_checkout(root, native_root, gguf_root=None, templates_root=None):
    bootstrap = ast.parse((root / "ruinedfooocus/app/bootstrap.py").read_bytes())
    declarations = {
        row["name"]: row for row in expression(assignments(bootstrap.body)["git_repos"])
    }
    expected = declarations["ComfyUI"]["hash"]
    actual = checked_commit(native_root, "ComfyUI", expected)
    gguf_declaration = declarations[NATIVE_REPOSITORIES["gguf"]]
    gguf_root = gguf_root or root / "repositories" / gguf_declaration["path"]
    gguf_pin = gguf_declaration["hash"]
    gguf_actual = checked_commit(gguf_root, NATIVE_REPOSITORIES["gguf"], gguf_pin)
    # Determine the audited extras from the pinned blob rather than a directory
    # scan or a potentially modified loader's file list.
    core = ast.parse(
        subprocess.check_output(
            ["git", "-C", str(native_root), "show", f"{actual}:nodes.py"], timeout=20
        )
    )
    roots = {"native": native_root, "gguf": gguf_root}
    locations = {
        "native/supported_models.py": native_root / "comfy/supported_models.py",
        "native/supported_models_base.py": native_root / "comfy/supported_models_base.py",
        "native/sd.py": native_root / "comfy/sd.py",
        "native/utils.py": native_root / "comfy/utils.py",
        "native/model_detection.py": native_root / "comfy/model_detection.py",
        "native/model_base.py": native_root / "comfy/model_base.py",
        "rf/image_components.py": root / "ruinedfooocus/features/models/image_components.py",
        "rf/image_pipeline.py": root / "ruinedfooocus/features/generation/image_pipeline.py",
        "rf/model_architecture.py": root / "ruinedfooocus/features/models/model_architecture.py",
        "rf/model_handler.py": root / "ruinedfooocus/features/models/model_handler.py",
    }
    locations.update(
        {key: roots[key.split("/", 1)[0]] / path for key, path in NATIVE_ANCHOR_PATHS.items()}
    )
    locations.update(
        {f"native/{path}": native_root / path for path in ["nodes.py", *builtin_node_paths(core)]}
    )
    shared_formats = root / "ruinedfooocus/core/model_formats.py"
    if shared_formats.exists():
        locations["rf/model_formats.py"] = shared_formats
    app_root = root / "ruinedfooocus"
    for caller in ("app/async_worker.py", "app/api_runtime.py", "integrations/openai_api.py"):
        locations["rf-callers/" + caller] = app_root / caller
    for directory in CALLER_DIRECTORIES:
        locations.update(
            {
                f"rf-callers/{path.relative_to(app_root).as_posix()}": path
                for path in sorted((app_root / directory).glob("*.py"))
            }
        )
    commits = {"native": actual, "gguf": gguf_actual}
    sources = {}
    for key, native_checkout in roots.items():
        paths = {
            path.relative_to(native_checkout).as_posix(): name
            for name, path in locations.items()
            if name.startswith(key + "/")
        }
        pinned = read_pin_sources(native_checkout, NATIVE_REPOSITORIES[key], commits[key], paths)
        sources.update({paths[path]: contents for path, contents in pinned.items()})
    sources.update(
        {name: path.read_bytes() for name, path in locations.items() if name not in sources}
    )
    input_hashes = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
    result = derive_sources(
        {
            name: sources[name]
            for name in (
                "native/supported_models.py",
                "native/supported_models_base.py",
                "native/sd.py",
                "native/utils.py",
                "rf/image_components.py",
                "rf/image_pipeline.py",
                "rf/model_architecture.py",
                "rf/model_handler.py",
                "rf/model_formats.py",
            )
            if name in sources
        }
    )
    result.update(derive_node_inventory(sources))
    result.update(
        schema=1,
        declared_native_pin=expected,
        inspected_native_commit=actual,
        native_audited_sources_match_pin=True,
        inspected_native_repositories={
            key: {
                "name": name,
                "declared_pin": declarations[name]["hash"],
                "commit": commits[key],
                "audited_sources_match_pin": True,
            }
            for key, name in NATIVE_REPOSITORIES.items()
        },
        input_sha256=input_hashes,
        source_sha256={str(path): input_hashes[name] for name, path in locations.items()},
        source_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        limits="AST/data only: config classes are not task families; registered native audio/video/3D/utility classes need their own callers. Routes are not runtime/semantic qualification. Custom nodes outside the pin are separate.",
    )
    resources = expression(
        assignments(
            ast.parse(
                (root / "ruinedfooocus/features/models/resource_defaults.py").read_bytes()
            ).body
        )["RESOURCE_DEFAULTS"]
    )
    for path in (
        root / "ruinedfooocus/app/bootstrap.py",
        root / "ruinedfooocus/features/models/resource_defaults.py",
    ):
        result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    downloads = {}
    for path in sorted((root / "ruinedfooocus/resources/model_catalogue").glob("*.json")):
        downloads.update(json.loads(path.read_bytes()))
        result["source_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    from ruinedfooocus.features.models.model_registry import registry_document

    metadata_root = root / "ruinedfooocus/resources/model_metadata"
    registry = registry_document(metadata_root)
    for registry_path in metadata_root.rglob("*.json"):
        result["source_sha256"][str(registry_path)] = hashlib.sha256(
            registry_path.read_bytes()
        ).hexdigest()
    result["registry_annotations"] = {
        "download_tuples": len(downloads),
        "profiles": len(registry.get("profiles", {})),
        "artifacts": len(registry.get("artifacts", {})),
        "role": "Exact tuple annotations, not a native architecture allowlist or retirement authority",
    }
    result["component_acquisition"] = {
        family: [
            {
                "role": key,
                "default_file": resources.get(key),
                "download_tuples": sorted(
                    name
                    for name, item in downloads.items()
                    if item.get("filename") == resources.get(key)
                    and item.get("path") == ("path_clip" if key.startswith("clip_") else "path_vae")
                ),
            }
            for key in list(clips) + ([] if vae == "pixel_space" else [vae])
        ]
        for family, (clips, vae) in result["RF_component_configurations"].items()
    }
    result["counts"] = {
        "native_registered_configs": len(result["native_config_classes"]),
        "native_clip_types": len(result["native_clip_types"]),
        "RF_runtime_configurations": len(result["RF_runtime_configurations"]),
        "RF_component_configurations": len(result["RF_component_configurations"]),
        "route_status": dict(
            Counter(row["route_status"] for row in result["native_config_classes"])
        ),
        "native_node_modules": len(result["native_node_modules"]),
        "native_registered_nodes": len(result["native_nodes"]),
        "native_nodes_by_registration": dict(
            Counter(f"{row['scope']}-{row['registration']}" for row in result["native_nodes"])
        ),
        "native_node_disposition": dict(
            Counter(row["disposition"] for row in result["native_nodes"])
        ),
        "operation_disposition": dict(
            Counter(row["disposition"] for row in result["operation_families"])
        ),
    }
    blueprint_paths = subprocess.check_output(
        ["git", "-C", str(native_root), "ls-tree", "-r", "--name-only", actual, "--", "blueprints"],
        text=True,
        timeout=20,
    ).splitlines()
    blueprint_paths = [path for path in blueprint_paths if path.endswith(".json")]
    blueprint_sources = read_pin_sources(native_root, "ComfyUI blueprints", actual, blueprint_paths)
    templates, template_identity = read_official_templates(templates_root)
    result["finite_inventory"] = reconcile_inventory(
        result, blueprint_sources, templates, template_identity
    )
    result["counts"].update(
        native_folder_roles=sum(
            "native_discovery_tokens" in row for row in result["native_folder_roles"]
        ),
        gguf_folder_aliases=sum(
            "native_base_roles" in row for row in result["native_folder_roles"]
        ),
        blueprints=result["finite_inventory"]["blueprints"]["counts"],
        official_templates=result["finite_inventory"]["official_templates"]["counts"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--gguf-root", type=Path)
    parser.add_argument(
        "--templates-root", type=Path, help="Installed official template JSON data directory"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    result = read_checkout(
        root,
        (args.native_root or root / "repositories/ComfyUI").resolve(),
        args.gguf_root.resolve() if args.gguf_root else None,
        args.templates_root,
    )
    if args.baseline:
        previous = json.loads(args.baseline.read_bytes())
        if previous.get("schema") != result["schema"]:
            raise ValueError("Coverage baseline schema differs; review migration before comparison")
        result["drift"] = compare_baseline(previous, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "counts": result["counts"], "drift": result.get("drift")}
        )
    )
    if args.fail_on_drift and result.get("drift"):
        return int(any(value for value in result["drift"].values() if isinstance(value, list)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

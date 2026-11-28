"""Reconcile finite native recipe/role inventories without importing runtimes."""

from collections import Counter
import hashlib
import json
from pathlib import Path


# These are work assignments, not a model allowlist or execution qualifications.
PACKETS = {
    "F": "Native model readers, role aliases and local bundle formats",
    "E": "Still-image precision, media import/export and advanced tensor files",
    "R": "Ordered image references and edit conditioning",
    "P": "ControlNet modes, model patches and patch lifecycle",
    "G": "Generation, staged sampling, inpaint and ordered layers",
    "U": "Image utilities, depth, masks, composition and advanced primitives",
    "A": "Exact-artifact native/app execution and visual acceptance",
}

# Native folder role -> native reader, application root category, packet, boundary.
ROLE_CONTRACTS = {
    "checkpoints": (
        "comfy.sd.load_checkpoint_guess_config",
        "checkpoints",
        "F",
        "Bundled components retain precedence.",
    ),
    "configs": (
        "RF bounded local YAML recipe",
        None,
        "F",
        "Data-only overrides; never import a named class.",
    ),
    "loras": (
        "comfy.sd.load_lora_for_models",
        "loras",
        "F",
        "Match MODEL/CLIP targets and signed strength.",
    ),
    "vae": (
        "comfy.sd.VAE",
        "vae",
        "F",
        "Match latent channels, spatial/temporal layout, alpha and codec.",
    ),
    "text_encoders": (
        "comfy.sd.load_clip",
        "clip",
        "F",
        "Match native CLIP type, encoder count, tokenizer and projector.",
    ),
    "diffusion_models": (
        "comfy.sd.load_diffusion_model",
        "diffusion_models",
        "F",
        "Require native architecture detection and compatible companions.",
    ),
    "clip_vision": (
        "comfy.clip_vision.load",
        "clip_vision",
        "R",
        "Still-image Style/Redux/PhotoMaker and UnCLIP have distinct dimensions.",
    ),
    "style_models": (
        "comfy.sd.load_style_model",
        "style_models",
        "R",
        "StyleAdapter and Redux are distinct readers/layouts.",
    ),
    "embeddings": (
        "comfy.sd1_clip.load_embed",
        "embeddings",
        "F",
        "Match the selected text encoder embedding width.",
    ),
    "diffusers": (
        "comfy.diffusers_load.load_diffusers",
        "diffusers",
        "F",
        "Only the captured local native single-file layout; shards need separate review.",
    ),
    "vae_approx": (
        "comfy.sd.VAE / RF preview decoder",
        "vae_approx",
        "U",
        "Preview decoders and full codecs are different roles.",
    ),
    "controlnet": (
        "comfy.controlnet.load_controlnet",
        "controlnet",
        "P",
        "Match base family, union mode and mask/guide contract.",
    ),
    "gligen": (
        "comfy.sd.load_gligen",
        "gligen",
        "U",
        "Captured ordered text boxes on compatible native SD UNet/CLIP and eight-times RGB codec.",
    ),
    "upscale_models": (
        "UpscaleModelLoader / Spandrel",
        "upscalers",
        "U",
        "Retain native scale, channel count and exact target geometry.",
    ),
    "latent_upscale_models": (
        "LatentUpscaleModelLoader",
        "latent_upscalers",
        None,
        "The reviewed fca baseline has seven blueprints and sixteen installed templates using temporal B,C,T,H,W Hunyuan/LTX/MiniMax latents and video codecs. Current named still adapters do not bind this role; RF consumes it in video refinement. This bounded workflow disposition does not claim that T=1 is impossible.",
    ),
    "custom_nodes": (
        "native custom-node loader",
        None,
        None,
        "Arbitrary third-party code is outside the authorized native pin.",
    ),
    "datasets": (
        "native dataset nodes",
        None,
        None,
        "Training datasets are outside still-image inference scope.",
    ),
    "hypernetworks": (
        "HypernetworkLoader",
        "hypernetworks",
        "U",
        "Captured native attention dimensions, signed strength and request-local native clone.",
    ),
    "photomaker": (
        "PhotoMakerLoader",
        "photomaker",
        "R",
        "SDXL trigger/token/vision contract; not arbitrary image adapters.",
    ),
    "classifiers": (
        "reserved native folder registration",
        None,
        None,
        "At the reviewed 0.45 pin this bucket is registered without a native reader or maintained consuming workflow; WeVisOne uses text_encoders instead. The registration alone creates no still-image operation.",
    ),
    "model_patches": (
        "ModelPatchLoader",
        "model_patches",
        "P",
        "Typed Qwen 2.1 Union, Qwen DiffSynth, Z-Image Fun, Anima LLLite and USO source routes; exact-artifact runtime acceptance stays separate.",
    ),
    "audio_encoders": (
        "native audio encoder readers",
        None,
        None,
        "Audio execution is outside still-image scope.",
    ),
    "background_removal": (
        "LoadBackgroundRemovalModel",
        "background_removal",
        "U",
        "Compatible BiRefNet source route; qualify alpha edges separately.",
    ),
    "frame_interpolation": (
        "native interpolation readers",
        None,
        None,
        "Video execution is explicitly skipped.",
    ),
    "geometry_estimation": (
        "LoadDA3Model / LoadMoGeModel",
        "geometry_estimation",
        "U",
        "Still depth/normal maps apply; mesh export remains outside scope.",
    ),
    "optical_flow": (
        "native optical-flow readers",
        None,
        None,
        "Temporal/video operations are outside still-image scope.",
    ),
    "detection": (
        "LoadMediaPipeFaceLandmarker",
        "detection",
        "U",
        "Still face landmark overlays use the complete MediaPipe bundle; this does not qualify mesh or video execution.",
    ),
    "unet_gguf": (
        "gguf_sd_loader",
        "diffusion_models",
        "F",
        "Image architecture and quantization checks remain native.",
    ),
    "clip_gguf": (
        "gguf_clip_loader",
        "clip",
        "F",
        "Select by case-insensitive container plus native encoder layout.",
    ),
    "lora_gguf": (
        "load_gguf_lora",
        "loras",
        "F",
        "Connected explicit image LoRA path; exact native target validation required.",
    ),
    "vae_gguf": (
        "VAELoaderGGUF",
        None,
        None,
        "Pin accepts only MiniMax H3 Q8_CR video VAE; no native still-image codec.",
    ),
    "latent_upscale_models_gguf": (
        "LTXVLatentUpscaleModelLoaderGGUF",
        None,
        None,
        "Pinned GGUF latent upscaler is video-only.",
    ),
}

NODE_PACKETS = {
    "SaveImageAdvanced": "E",
    "SaveImage": "E",
    "LoadImage": "E",
    "LoadImageMask": "E",
    "LoadImageOutput": "E",
    "SaveConditioning": "E",
    "ConditioningLoader": "E",
    "SaveLatent": "E",
    "LoadLatent": "E",
    "PreviewImage": "E",
    "ReferenceLatent": "R",
    "ReferenceMultiLatent": "R",
    "FluxKontextMultiReferenceLatentMethod": "R",
    "TextEncodeQwenImageEditPlus": "R",
    "TextEncodeBooguEdit": "R",
    "TextEncodeMingImageEdit": "R",
    "TextEncodeZImageOmni": "R",
    "EmptyQwenImageLayeredLatentImage": "G",
    "SetUnionControlNetType": "P",
    "AnimaLLLiteApply": "P",
    "USOStyleReference": "P",
    "QwenImageDiffsynthControlnet": "P",
    "ZImageFunControlnet": "P",
    "ControlNetInpaintingAliMamaApply": "P",
    "CreateHookLora": "P",
    "CreateHookLoraModelOnly": "P",
    "CreateHookKeyframe": "P",
    "CreateHookKeyframesInterpolated": "P",
    "SetHookKeyframes": "P",
    "SetModelHooksOnCond": "P",
    "SetClipHooks": "P",
    "CreateHookModelAsLora": "P",
    "CreateHookModelAsLoraModelOnly": "P",
    "CreateHookKeyframesFromFloats": "P",
    "CombineHooks2": "P",
    "CombineHooks4": "P",
    "CombineHooks8": "P",
}

# Category/module names describe native organization, not an operation's media
# contract. These bounded exceptions retain still consumers of shared runtimes.
SHARED_IMAGE_NODE_PACKETS = {
    "HunyuanRefinerLatent": "G",
    "MiniMaxH3ImageConditioning": "G",
    "SeedVR2Preprocess": "U",
    "SeedVR2PostProcessing": "U",
    "SeedVR2Conditioning": "U",
    "LTXVSeparateAVLatent": "U",
}
IMAGE_SCOPE_EXCLUSIONS = {
    "Kandinsky5ImageToVideo": "video",
    "NormalizeVideoLatentStart": "video",
    "ReplaceVideoLatentFrames": "video",
    "UNetTemporalAttentionMultiply": "video",
    "TemporalScoreRescaling": "video",
    "ContextWindowsManual": "video",
    "MiniMaxH3ContextWindows": "video",
    "ModelMergeCosmos14B": "video",
    "ModelMergeCosmos7B": "video",
    "ModelMergeCosmosPredict2_14B": "video",
    "ModelMergeCosmosPredict2_2B": "video",
    "ModelMergeLTXV": "video",
    "ModelMergeMochiPreview": "video",
    "ModelMergeWAN2_1": "video",
    "wanBlockSwap": "video",
    "LatentUpscaleModelLoader": "video",
    "MoGePointMapToMesh": "3d",
}


def image_node_domain(row):
    if row["id"] in SHARED_IMAGE_NODE_PACKETS:
        return "image"
    if row["id"] in IMAGE_SCOPE_EXCLUSIONS:
        return IMAGE_SCOPE_EXCLUSIONS[row["id"]]
    if row["native_domain"] == "training" and row.get("native_category") in {
        "image/transform",
        "image/color",
        "image/batch",
        "text",
    }:
        return row["native_category"].split("/", 1)[0]
    return row["native_domain"]


def assignment(packet, disposition, reason):
    return {
        "owner": "W07",
        "packet": packet,
        "disposition": disposition,
        "reason": reason,
        "evidence": {
            "source": "inspected",
            "ui_api": "pending",
            "real_runtime": "pending",
            "visual_quality": "pending",
        },
    }


def node_assignment(row):
    domain = image_node_domain(row)
    if domain in {"audio", "video", "3d", "training"}:
        return assignment(
            None, "scoped-exclusion", f"{domain} execution is outside this still-image plan."
        )
    node_id, role = row["id"], row["role"]
    packet = SHARED_IMAGE_NODE_PACKETS.get(node_id, NODE_PACKETS.get(node_id))
    if packet is None:
        if role.endswith("loader") or role in {
            "lora-import",
            "quantization-tool",
            "model-artifact",
        }:
            packet = "F"
        elif "reference" in node_id.casefold() or "Edit" in node_id:
            packet = "R"
        elif role in {"model-patch", "guidance", "hook"} or "ControlNet" in node_id:
            packet = "P"
        elif role in {"latent", "sampling", "conditioning"}:
            packet = "G"
        else:
            packet = "U"
    disposition = (
        "source-consumer-needs-acceptance"
        if row["RF_source_references"]
        else "planned-operation-review"
    )
    return assignment(
        packet,
        disposition,
        "Source references and graph primitives do not establish a complete UI or runtime operation.",
    )


def config_assignment(row):
    name = row["class"]
    if name in {"MiniMaxH3", "SeedVR2", "HunyuanVideo15", "WAN21_T2V", "WAN21_I2V"}:
        return assignment(
            "U" if name == "SeedVR2" else "G",
            "planned-operation-review",
            "Official still-image workflow exists for this shared video-capable class; the image route remains unfinished independently of video scope.",
        )
    if name in {"Stable_Zero123", "SV3D_u", "SV3D_p", "TripoSplat", "Trellis2"} or name.startswith(
        "Hunyuan3D"
    ):
        return assignment(
            None, "scoped-exclusion", "3D/view synthesis is outside still-image inference scope."
        )
    if name.startswith(("StableAudio", "ACEStep", "MiniMaxMusic", "YuE")):
        return assignment(None, "scoped-exclusion", "Audio execution is outside still-image scope.")
    if name.startswith(
        ("WAN", "HunyuanVideo", "CogVideo", "LTX", "CosmosI2V", "CosmosT2V", "SVD_")
    ) or name in {"Kandinsky5", "GenmoMochi", "MiniMaxH3"}:
        return assignment(
            None,
            "scoped-exclusion",
            "Video execution is explicitly skipped; image variants retain separate rows.",
        )
    if name in {"RT_DETR_v4", "SAM3", "SAM31", "DepthAnything3", "LotusD", "SeedVR2"}:
        packet = "U"
    else:
        packet = "G"
    disposition = (
        "source-consumer-needs-acceptance"
        if row["RF_runtime_classes"]
        else "planned-operation-review"
    )
    return assignment(
        packet,
        disposition,
        "Specialized auxiliary consumers may exist independently of ordinary generation registration.",
    )


def recipe_graphs(document, location="root"):
    """Yield graph definitions once; graph-call wrappers are not native node IDs."""
    if isinstance(document, dict) and isinstance(document.get("nodes"), list):
        yield location, document
    for index, graph in enumerate(document.get("definitions", {}).get("subgraphs", [])):
        yield from recipe_graphs(graph, f"{location}/subgraphs/{index}")


def recipe_inventory(documents, native_nodes, *, official=False):
    nodes = {row["id"]: row for row in native_nodes}
    metadata = {}
    if official and "index.json" in documents:
        index = json.loads(documents["index.json"])
        if isinstance(index, list):
            for group in index:
                for template in group.get("templates", []):
                    metadata[template["name"] + ".json"] = {
                        "type": group.get("type"),
                        "media_type": template.get("mediaType"),
                        "tags": template.get("tags", []),
                    }
    recipes, graph_nodes, model_slots = [], {}, []
    for filename, raw in sorted(documents.items()):
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
            continue  # Index/schema assets are recorded separately, never counted as recipes.
        graphs = list(recipe_graphs(document))
        graph_ids = {graph.get("id") for _, graph in graphs} - {None}
        leaf_ids, wrappers, slots, leaf_count = set(), 0, [], 0
        for location, graph in graphs:
            for node in graph["nodes"]:
                node_id = node.get("type")
                if not isinstance(node_id, str):
                    raise ValueError(f"{filename}: graph node type needs review")
                if node_id in graph_ids:
                    wrappers += 1
                    continue
                leaf_ids.add(node_id)
                leaf_count += 1
                if node_id not in graph_nodes:
                    native = nodes.get(node_id)
                    graph_nodes[node_id] = {
                        "id": node_id,
                        "native_registered": native is not None,
                        **(
                            node_assignment(native)
                            if native
                            else assignment(
                                "U",
                                "external-or-frontend-node-review",
                                "Not in the pinned native registry; identify frontend helpers, hosted API and third-party nodes before use.",
                            )
                        ),
                    }
                for model in node.get("properties", {}).get("models", []):
                    if not isinstance(model, dict) or not all(
                        isinstance(model.get(k), str) for k in ("name", "directory", "url")
                    ):
                        raise ValueError(f"{filename}: model source annotation needs review")
                    slot = {
                        "recipe": filename,
                        "graph": location,
                        "node_instance": node.get("id"),
                        "node_id": node_id,
                        **model,
                    }
                    contract = ROLE_CONTRACTS.get(model["directory"])
                    slot.update(
                        assignment(
                            contract[2] if contract else "F",
                            "source-artifact-needs-role-validation",
                            "A native recipe annotation is provenance, not verified bytes, compatibility or acquisition authorization.",
                        )
                    )
                    slots.append(slot)
        lower = Path(filename).stem.casefold()
        info = metadata.get(Path(filename).name, {})
        media = info.get("type") or info.get("media_type")
        domains = {image_node_domain(nodes[node_id]) for node_id in leaf_ids if node_id in nodes}
        excluded_nodes = sorted(
            node_id
            for node_id in leaf_ids
            if node_id in nodes
            and image_node_domain(nodes[node_id]) in {"audio", "video", "3d", "training"}
        )
        root_calls = {node["type"] for node in document["nodes"]}
        interface_graphs = (
            [document]
            if "outputs" in document
            else [graph for _, graph in graphs if graph.get("id") in root_calls]
        )
        declared_output_types = sorted(
            {
                port["type"]
                for graph in interface_graphs
                for port in graph.get("outputs", [])
                if isinstance(port, dict) and isinstance(port.get("type"), str)
            }
        )
        graph_video = bool(
            leaf_ids & {"SaveVideo", "CreateVideo", "LoadVideo", "GetVideoComponents"}
        )
        graph_image = bool(leaf_ids & {"SaveImage", "SaveImageAdvanced"}) and not graph_video
        graph_3d = "3d" in domains or bool(
            leaf_ids
            & {
                "SaveGLB",
                "SaveOBJ",
                "SaveMesh",
                "VAEDecodeTripoSplat",
                "TripoSplatConditioning",
                "VoxelToMesh",
            }
        )
        graph_audio = (
            bool(
                leaf_ids
                & {
                    "VAEDecodeAudio",
                    "VAEDecodeAudioTiled",
                    "SaveAudio",
                    "SaveAudioMP3",
                    "SaveAudioOpus",
                    "YuE2GenerateMusic",
                }
            )
            and not graph_video
            and not graph_image
        )
        if lower.startswith("api_") or "API" in info.get("tags", []) or "GeminiNode" in leaf_ids:
            scope = "hosted-partner"
        elif graph_3d and not graph_image and "IMAGE" not in declared_output_types:
            scope = "3d"
        elif graph_video:
            scope = "video"
        elif graph_image:
            scope = "still-image-or-utility"
        elif graph_audio:
            scope = "audio"
        elif (
            media == "3d"
            or lower.startswith(("3d_", "3d "))
            or any(word in lower for word in ("to mesh", "to model", "point cloud"))
        ):
            scope = "3d"
        elif (
            media == "audio"
            or lower.startswith(("audio", "music"))
            or any(word in lower for word in ("to audio", "to music"))
        ):
            scope = "audio"
        elif (
            media == "video"
            or lower.startswith("video")
            or any(
                word in lower for word in ("to video", "video ", "interpolation", "to animation")
            )
            or (not official and "video" in domains)
        ):
            scope = "video"
        else:
            scope = "still-image-or-utility"
        recipes.append(
            {
                "id": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "scope": scope,
                "root_graphs": 1,
                "nested_graphs": len(graphs) - 1,
                "subgraph_wrappers": wrappers,
                "leaf_node_instances": leaf_count,
                "node_ids": sorted(leaf_ids),
                "excluded_node_ids": excluded_nodes,
                "shared_image_node_ids": sorted(leaf_ids & SHARED_IMAGE_NODE_PACKETS.keys()),
                "declared_output_types": declared_output_types,
                "model_slot_count": len(slots),
                **assignment(
                    None if scope != "still-image-or-utility" else "U",
                    "scoped-exclusion"
                    if scope != "still-image-or-utility"
                    else "planned-operation-review",
                    "Recipe presence does not establish an RF UI route or exact-artifact execution.",
                ),
            }
        )
        model_slots.extend(slots)
    counts = {
        "recipes": len(recipes),
        "nested_graphs": sum(r["nested_graphs"] for r in recipes),
        "subgraph_wrappers": sum(r["subgraph_wrappers"] for r in recipes),
        "leaf_node_instances": sum(r["leaf_node_instances"] for r in recipes),
        "node_ids": len(graph_nodes),
        "model_slots": len(model_slots),
        "scopes": dict(Counter(r["scope"] for r in recipes)),
    }
    return {
        "kind": "installed-official-templates" if official else "pinned-native-blueprints",
        "counts": counts,
        "recipes": recipes,
        "node_ids": sorted(graph_nodes.values(), key=lambda r: r["id"]),
        "model_slots": model_slots,
    }


def reconcile_inventory(report, blueprints, templates, template_identity):
    roles = []
    for row in report["native_folder_roles"]:
        if row["role"] not in ROLE_CONTRACTS:
            raise ValueError(f"Native role {row['role']!r} has no reviewed packet contract")
        reader, category, packet, boundary = ROLE_CONTRACTS[row["role"]]
        roles.append(
            {
                **row,
                "native_reader": reader,
                "RF_root_category": category,
                **assignment(
                    packet,
                    "scoped-exclusion"
                    if packet is None
                    else "source-role-needs-consumer-validation",
                    boundary,
                ),
            }
        )
    clips = []
    image_clips = set(report.get("RF_image_clip_types", [])) | {
        "MINIMAX",
        "HUNYUAN_VIDEO_15",
        "WAN",
    }
    video_clips = {
        "MOCHI",
        "LTXV",
        "HUNYUAN_VIDEO",
        "HUNYUAN_VIDEO_15",
        "COSMOS",
        "WAN",
        "KANDINSKY5",
        "COGVIDEOX",
        "MINIMAX",
    }
    audio_clips = {"STABLE_AUDIO", "ACE", "YUE2"}
    for name, value in report["native_clip_types"].items():
        excluded = name in video_clips | audio_clips and name not in image_clips
        clips.append(
            {
                "id": name,
                "native_value": value,
                **assignment(
                    None if excluded else "F",
                    "scoped-exclusion" if excluded else "source-role-needs-consumer-validation",
                    "Native CLIP enum entry alone does not prove a compatible encoder file or workflow.",
                ),
            }
        )
    return {
        "schema_version": 1,
        "owner": "W07",
        "packets": PACKETS,
        "policy": "All rows have an explicit current disposition and owner. Planned review is unfinished work; source, UI/API, exact runtime and visual acceptance remain independent. No implementation absence is reclassified as unsupported.",
        "native_nodes": [
            {"id": row["id"], **node_assignment(row)} for row in report["native_nodes"]
        ],
        "native_configs": [
            {"id": row["class"], **config_assignment(row)}
            for row in report["native_config_classes"]
        ],
        "native_clip_types": clips,
        "native_roles": roles,
        "blueprints": recipe_inventory(blueprints, report["native_nodes"]),
        "official_templates": {
            **recipe_inventory(templates, report["native_nodes"], official=True),
            "identity": template_identity,
        },
        "W02_contract": {
            "reader_role_before_suffix": True,
            "unknown_files_remain_visible": True,
            "identity_fields": ["canonical_path", "size", "mtime_ns", "ctime_ns"],
            "optional_identity_fields": [
                "sha256",
                "source_revision",
                "parser_version",
                "runtime",
                "provider",
                "profile_scope",
            ],
            "cache_envelope": {
                "schema_version": 1,
                "namespace": "native-role-metadata",
                "identity": "file identity plus parser/source revision",
                "payload": "separate architecture, variant, component roles, recommendations and artifact qualification",
            },
            "required_role_fields": [
                "native_reader",
                "container",
                "dtype",
                "logical_shape",
                "native_family",
                "component_role",
                "companions",
                "publisher_variant",
                "recommendation_source",
                "acquisition_tuple",
                "qualification_receipts",
            ],
            "constraints": [
                "No downloads from recipe annotations without artifact validation.",
                "Bundled components retain precedence; explicit compatible user overrides retain precedence over defaults.",
                "GGUF VAE is video-only at this pin; image VAE codecs use native state dictionaries.",
                "An unregistered model or alternate quantization can use recognized family defaults without exact-artifact qualification.",
                "Model-patch files require their distinct native patch class; generic ControlNet compatibility is insufficient.",
                "Keep still image, video, audio and 3D artifact categories separate.",
            ],
        },
    }

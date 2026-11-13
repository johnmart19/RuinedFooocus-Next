"""Finite MoGe render choices and topology-owned face selection controls."""

import gradio as gr
from ruinedfooocus.core.translation_manager import translate_ui as t


def moge_controls(saved):
    controls = {}
    with gr.Row():
        controls["moge_output"] = gr.Dropdown(
            label=t("MoGe output"),
            choices=[(t("Depth"), "depth"), (t("Surface normals"), "normal")],
            value=saved.get("moge_output", "depth"),
        )
        controls["moge_projection"] = gr.Dropdown(
            label=t("Source projection"),
            choices=[(t("Perspective"), "perspective"), (t("360° panorama depth"), "panorama")],
            value=saved.get("moge_projection", "perspective"),
        )
    controls["moge_normal_convention"] = gr.Dropdown(
        label=t("Normal convention"),
        choices=[("OpenGL", "opengl"), ("DirectX", "directx")],
        value=saved.get("moge_normal_convention", "opengl"),
        visible=saved.get("moge_output") == "normal",
    )
    with gr.Accordion(
        t("Panorama settings"), open=False, visible=saved.get("moge_projection") == "panorama"
    ) as panorama:
        controls["moge_split_resolution"] = gr.Slider(
            label=t("Panorama view resolution"),
            minimum=256,
            maximum=1024,
            step=1,
            value=saved.get("moge_split_resolution", 512),
        )
        controls["moge_merge_resolution"] = gr.Slider(
            label=t("Panorama merge resolution"),
            minimum=256,
            maximum=8192,
            step=1,
            value=saved.get("moge_merge_resolution", 1920),
        )
        controls["moge_panorama_batch_size"] = gr.Slider(
            label=t("Panorama views per batch"),
            minimum=1,
            maximum=12,
            step=1,
            value=saved.get("moge_panorama_batch_size", 1),
        )

    def moge_output_context(output, projection):
        pano = projection == "panorama"
        choices = [(t("Depth"), "depth")] + ([] if pano else [(t("Surface normals"), "normal")])
        return (
            gr.update(choices=choices, **({"value": "depth"} if pano else {})),
            gr.update(visible=output == "normal" and not pano),
            gr.update(visible=pano),
        )

    gr.on(
        triggers=[controls["moge_output"].change, controls["moge_projection"].change],
        fn=moge_output_context,
        inputs=[controls["moge_output"], controls["moge_projection"]],
        outputs=[controls["moge_output"], controls["moge_normal_convention"], panorama],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return controls


def face_controls(saved):
    controls = {}
    controls["face_output"] = gr.Dropdown(
        label=t("Face output"),
        choices=[(t("Landmark overlay"), "overlay"), (t("Face region mask"), "region-mask")],
        value=saved.get("face_output", "overlay"),
    )
    with gr.Group(visible=saved.get("face_output") == "region-mask") as masks:
        controls["face_mask_regions"] = gr.Radio(
            label=t("Face mask regions"),
            choices=[(t("All face regions"), "all"), (t("Custom regions"), "custom")],
            value=saved.get("face_mask_regions", "all"),
        )
        with gr.Group(visible=saved.get("face_mask_regions") == "custom") as custom:
            for name, label, default in (
                ("face_oval", "Face oval", True),
                ("lips", "Lips", False),
                ("left_eye", "Left eye", False),
                ("right_eye", "Right eye", False),
                ("irises", "Irises", False),
            ):
                key = "face_mask_" + name
                controls[key] = gr.Checkbox(label=t(label), value=saved.get(key, default))

    def face_output_context(output, regions):
        return gr.update(visible=output == "region-mask"), gr.update(visible=regions == "custom")

    gr.on(
        triggers=[controls["face_output"].change, controls["face_mask_regions"].change],
        fn=face_output_context,
        inputs=[controls["face_output"], controls["face_mask_regions"]],
        outputs=[masks, custom],
        queue=False,
        show_progress="hidden",
        api_visibility="undocumented",
    )
    return controls

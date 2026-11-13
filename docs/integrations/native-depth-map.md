# Standalone Depth Anything 3 maps

In **Create → Image tools**, choose **Depth map**, add a source image, and select
an installed **Depth Anything 3 model**. The compatible models come from the
configured `geometry_estimation` roots. The Models catalogue already offers
DA3 Small and Base. Once the model is installed, this tool can run offline.

**Generate** produces one 8-bit grayscale PNG at the source's EXIF-oriented
dimensions. It uses the source RGB channels. White represents near surfaces.
The relative depth map can be downloaded from the result or used as the supplied
depth map in an existing Depth ControlNet workflow. Library output records the
model basename and size, inference resolution, resize method, and normalization.
To rerun estimation, supply the original image and select the installed model.

The tool uses the native vision worker and its shared compute admission, model
cache, cancellation, and read leases. Capture validates the DA3 architecture
from the safetensors header and freezes the file identity and source image.
The pinned `DA3Inference` and `DA3Render` run in monocular mode with a 504-pixel
long-side limit and V2-style normalization. The worker checks output size and
finite values before writing the PNG. The route requires a DA3 artifact and
source image. It does not require diffusion sampling, a generation checkpoint,
text encoders, a VAE, LoRA, or downloads during submission.

The API also accepts `controlnet: {"type":"depth-map"}` and
`native_vision_model: "depth_anything_3_small.safetensors"` on
`POST /v1/images/generations` with `input_image`. For multipart
`POST /v1/images/edits`, send the source as `image` and the `controlnet` field as
JSON text. The model filename is relative to a configured `geometry_estimation`
root. Use one input and one result per request. Leave `size` unset, use `auto`,
or supply the exact oriented source size. Use `response_format: "b64_json"` to
receive the PNG inline or `response_format: "url"` to retrieve it separately.

The native source pin is `fca60fe70e5b84c5c6b058263653d15cb3401e44`, in
`comfy_extras/nodes_depth_anything_3.py`. DA3 ControlNet and Lotus remain separate
operations. Local acceptance should check a non-square photo, native/app pixel
agreement, orientation, Library download, cancellation, and offline operation
with an installed DA3 model on the selected device.

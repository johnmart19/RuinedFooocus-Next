# Portable owner media paths

The owner UI can keep mutable data outside the application directory through
`RF_DATA_DIR` and configured model/output paths. Gradio still checks callback
file outputs before copying them into its own cache. The launch allowlist must
therefore include the dedicated media folders used by those callbacks.

`integrations.media_paths.owner_media_paths` retains the existing asset, output,
model-preview cache and explicit archive roots. It adds only five leaves of the
configured cache: `card-previews`, `catalogue-previews`, `default-previews`,
`native-image-previews` and `native-image-exports`. The labelled neutral model
artwork uses `default-previews`, including both the card image and browser decode
fallback returned through Gradio's HTML/FileData serializer after launch.
It does not allow the whole data/cache tree, settings,
credentials, model payload folders or hash records. A redirected media leaf
resolving outside the configured cache is excluded.

The launch binding uses that list as `allowed_paths`. Gradio's native file check,
uploaded-input restriction, owner login and private-response middleware remain
active. The guest launch keeps its empty allowlist and checked guest-media
tokens; its generic file/upload routes stay blocked.

```sh
python tools/check_cpu.py test_media_paths.py test_api_live_contracts.py test_guest_localization.py
```

The focused CPU tests reproduce the missing-root exception in a real Gradio
callback with separate source/data/temp paths, then transport images from each
owned leaf, including actual category-default creation under an external
`RF_DATA_DIR` and the launched model-card callback serializer. They also cover authenticated HTTP file access, private settings
refusal, unuploaded-input refusal and guest blocking. They do not qualify a live
browser or model execution.

# Model folders

See the [folder guide](../docs/model-folders.md) for supported image-model folders,
legacy lookup and profile overrides. Place new encoders in `text_encoders`,
diffusion-only weights in `diffusion_models`, and textual-inversion embeddings in
`embeddings`. Placeholder files are small instructions, not bundled weights.
Startup does not move existing files or recreate legacy alias folders. The guide
describes explicit consolidation with collision checks and weight verification.

# Model folders

See the [folder guide](../docs/model-folders.md) for supported image-model folders,
legacy lookup and profile overrides. Place new encoders in `text_encoders`,
diffusion-only weights in `diffusion_models`, and textual-inversion embeddings in
`embeddings`. Chat retrieval encoders use `rag_embeddings`.
No model weights or placement marker files are bundled. Startup and Recheck folders
create missing default image/Chat model directories. Actual scans and first-use
downloads also recover default categories after removal. Missing custom model
locations must be restored by their owner. Existing weights are never moved, and
legacy alias folders are not recreated. The guide
describes explicit consolidation with collision checks and weight verification.

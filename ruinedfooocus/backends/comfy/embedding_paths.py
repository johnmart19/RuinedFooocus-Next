"""Register configured text-embedding roots at the native model-load boundary."""

from pathlib import Path


def register_embedding_paths(directories):
    import folder_paths

    # Native GGUF text loaders read this registry themselves. Other native
    # loaders accept the same ordered roots explicitly. Keep native defaults.
    for directory in reversed(directories):
        folder_paths.add_model_folder_path(
            "embeddings", str(Path(directory).resolve()), is_default=True
        )
    return folder_paths.get_folder_paths("embeddings")

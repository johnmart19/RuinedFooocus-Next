from gradio.processing_utils import get_upload_folder, save_file_to_cache
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import json
from typing import Dict
from pathlib import Path
from ruinedfooocus.app.state import settings, path_manager


def format_metadata(metadata: Dict) -> Dict:
    """Format metadata into a more readable structure."""
    try:
        # Create formatted output dictionary
        formatted = {"File Path": metadata.get("file_path", "Unknown")}
        formatted["Settings"] = {
            key.capitalize(): metadata[key]
            for key in ("width", "height", "frames")
            if key in metadata
        }

        # Parse the parameters string if it exists
        if "parameters" in metadata:
            params = json.loads(metadata["parameters"])

            # Add generation parameters in a logical order
            if "Prompt" in params:
                formatted["Prompt"] = params["Prompt"].strip()
            if "Negative" in params:
                formatted["Negative Prompt"] = params["Negative"].strip()

            # Model information
            if "base_model_name" in params:
                formatted["Model"] = params["base_model_name"]
            if "base_model_hash" in params:
                formatted["Model Hash"] = params["base_model_hash"]

            # Generation settings
            settings = {}
            for key in [
                "steps",
                "cfg",
                "width",
                "height",
                "seed",
                "sampler_name",
                "scheduler",
                "clip_skip",
                "denoise",
            ]:
                if key in params and params[key] is not None:
                    settings[key.capitalize()] = params[key]
            formatted["Settings"].update(settings)

            if params.get("generation_width") and params.get("generation_height"):
                formatted["Generation size"] = (
                    f"{params['generation_width']} x {params['generation_height']}"
                )
            if params.get("width") and params.get("height"):
                formatted["Output size"] = f"{params['width']} x {params['height']}"
            if params.get("upscaler"):
                formatted["Upscale"] = {
                    "Model": params["upscaler"],
                    "Option": params.get("automatic_upscale", "Input image upscale"),
                }

            # Comment info
            if "meta_comment" in params:
                formatted["Meta_Comment"] = params["meta_comment"]

            # Software info
            if "software" in params:
                formatted["Software"] = params["software"]

            # LoRAs if present
            if "loras" in params and params["loras"]:
                formatted["LoRAs"] = params["loras"]

        return formatted
    except Exception as e:
        print(f"Error formatting metadata: {e}")
        return metadata


def format_metadata_string(metadata: Dict) -> str:
    """Convert formatted metadata into a readable string."""
    try:
        formatted = format_metadata(metadata)
        output = []

        # File path
        output.append(f"File: {formatted['File Path']}\n")

        # Prompt
        if "Prompt" in formatted:
            output.append("Prompt:")
            output.append(formatted["Prompt"])
            output.append("")

        # Negative prompt
        if "Negative Prompt" in formatted and formatted["Negative Prompt"]:
            output.append("Negative Prompt:")
            output.append(formatted["Negative Prompt"])
            output.append("")

        # Model info
        if "Model" in formatted:
            output.append(f"Model: {formatted['Model']}")
            if "Model Hash" in formatted:
                output.append(f"Hash: {formatted['Model Hash']}")
            output.append("")

        # Settings
        if "Settings" in formatted:
            output.append("Generation Settings:")
            for key, value in formatted["Settings"].items():
                output.append(f"  {key}: {value}")
            output.append("")

        for key in ("Generation size", "Output size"):
            if key in formatted:
                output.append(f"{key}: {formatted[key]}")
        if "Upscale" in formatted:
            output.append(
                f"Upscale: {formatted['Upscale']['Option']} ({formatted['Upscale']['Model']})"
            )

        # Comment
        if "Meta_Comment" in formatted:
            output.append(f"Comment: {formatted['Meta_Comment']}")

        # Software
        if "Software" in formatted:
            output.append(f"Software: {formatted['Software']}")

        # LoRAs
        if "LoRAs" in formatted and formatted["LoRAs"]:
            output.append("\nLoRAs:")
            for lora in formatted["LoRAs"]:
                output.append(f"  {lora}")

        return "\n".join(output)
    except Exception as e:
        return f"Error formatting metadata string: {e}\n\nRaw metadata:\n{json.dumps(metadata, indent=2)}"


def get_png_metadata(image_path: str) -> Dict:
    """Extract metadata from PNG file."""
    with Image.open(image_path) as img:
        if getattr(img, "n_frames", 1) != 1:
            raise ValueError("Choose a single still image.")
        metadata = {}
        if isinstance(img, PngImageFile):
            for key, value in img.info.items():
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        value = str(value)
                metadata[key] = value
        img.verify()
        return metadata


def connect_database(path=None):
    from ruinedfooocus.features.gallery.index_store import connect_index

    if path is None:
        from ruinedfooocus.core.paths import data_path

        path = data_path("cache", "images.db")
    return connect_index(path)


def get_media_metadata(path):
    from ruinedfooocus.features.models.image_scope import require_image_path

    require_image_path(path)
    if path.suffix.lower() in (".exr", ".avif"):
        from ruinedfooocus.features.image_processing.native_image_formats import read_image

        imported = read_image(path)
        metadata = dict(
            imported.metadata,
            width=imported.pixels.shape[1],
            height=imported.pixels.shape[0],
            frames=1,
            bit_depth=imported.bit_depth,
            color_space=imported.color_space,
        )
        parameters = metadata.get("parameters")
    else:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Choose a single still image.")
            metadata = {
                "width": image.width,
                "height": image.height,
                "frames": getattr(image, "n_frames", 1),
            }
            parameters = image.info.get("parameters") or image.info.get("comment")
    if isinstance(parameters, bytes):
        parameters = parameters.decode("utf-8", errors="replace")
    if parameters:
        try:
            data = json.loads(parameters)
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                metadata["parameters"] = json.dumps(data)
        except (ValueError, TypeError):
            pass
    return metadata


class ImageBrowser:
    def __init__(self):
        from ruinedfooocus.core.cache_contract import configured_cache_root

        self.path_manager = path_manager
        self.base_path = Path(self.path_manager.model_paths["temp_outputs_path"])
        self._database_path = configured_cache_root(self.path_manager) / "images.db"
        self.sql_conn = connect_database(self._database_path)
        self.images_per_page = min(
            100, max(1, int(settings.default_settings.get("images_per_page", 100)))
        )

    def num_images_pages(self, search=""):
        from ruinedfooocus.features.gallery.index_store import index_connection, IMAGE_ROWS

        with index_connection(self) as connection:
            image_cnt = connection.execute(
                f"SELECT count(*) FROM images WHERE json LIKE ? AND {IMAGE_ROWS}",
                (f"%{search or ''}%",),
            ).fetchone()[0]
        pages = max(1, (image_cnt + self.images_per_page - 1) // self.images_per_page)
        return image_cnt, pages

    def load_images(
        self,
        page: int,
        search="",
        *,
        include_paths=False,
        locale=None,
        cancelled=None,
        include_page=False,
        page_size=None,
    ):
        from ruinedfooocus.core.translation_manager import localize
        from ruinedfooocus.features.gallery.index_store import (
            check_cancelled,
            index_connection,
            page_snapshot,
        )

        check_cancelled(cancelled)
        size = self.images_per_page if page_size is None else min(100, max(1, int(page_size)))
        with index_connection(self) as connection:
            count, pages, page, rows = page_snapshot(connection, page, search, size)
        image_paths = []
        cached_paths = []
        records = []
        from contextlib import nullcontext
        import os
        from ruinedfooocus.core.cache_contract import configured_cache_root
        from ruinedfooocus.features.gallery.actions import source_record
        from ruinedfooocus.features.gallery.thumbnails import grid_thumbnail_batch

        manager = getattr(self, "path_manager", None)
        cache = configured_cache_root(manager) if manager is not None else None
        copies = (
            grid_thumbnail_batch(cache) if cache is not None else nullcontext(lambda path: path)
        )
        upload = get_upload_folder()
        previous_copies = getattr(self, "_page_copies", {})
        page_copies = {}
        with copies as thumbnail:
            for fullpath, relpath, *_ in rows:
                check_cancelled(cancelled)
                try:
                    record = source_record(fullpath)
                    from ruinedfooocus.features.image_processing.native_image_formats import (
                        display_image,
                        needs_preview,
                    )

                    display = (
                        display_image(fullpath, cache) if needs_preview(fullpath) else fullpath
                    )
                    display = thumbnail(display)
                    key = (str(upload), str(display), tuple(record["signature"]))
                    cached = previous_copies.get(key)
                    if cached is not None:
                        try:
                            if Path(cached).is_file():
                                # Match native Gradio's renewed delete_cache grace period.
                                os.utime(cached, None)
                            else:
                                cached = None
                        except OSError:
                            cached = None
                    if cached is None:
                        cached = save_file_to_cache(str(display), upload)
                    page_copies[key] = cached
                    cached_paths.append((cached, record["caption"]))
                    records.append(record)
                    image_paths.append((fullpath, relpath))
                except (OSError, ValueError):
                    # Files can disappear or be rejected between query and cache copy.
                    continue
        # Immutable replacement keeps concurrent readers independent and bounds
        # reuse to one page (at most 100 files); stale workers can only lose hits.
        self._page_copies = page_copies
        if image_paths:
            first = (page - 1) * size + 1
            last = first + len(image_paths) - 1
            text = (
                localize("Images: {count}", locale, count=count)
                if pages == 1
                else localize(
                    "Images {first}–{last} of {count}", locale, first=first, last=last, count=count
                )
            )
        else:
            text = localize("No images found", locale)

        result = (cached_paths, text)
        if include_page:
            return (*result, records, {"count": count, "pages": pages, "page": page})
        return (*result, records) if include_paths else result

    def add_image(self, full_path, rel_path, metadata, commit=False):
        if commit:
            self.sql_conn.cursor()
        self.sql_conn.execute(
            "INSERT INTO images (fullpath, path, json) VALUES (?, ?, ?)",
            (str(full_path), str(rel_path), json.dumps(metadata)),
        )
        if commit:
            self.sql_conn.commit()

    def update_images(
        self,
        locale=None,
        *,
        cancelled=None,
        progress=None,
        publish=None,
        only_if_uninitialized=False,
    ):
        """Refresh changed files without replacing the previous committed snapshot early."""
        import errno
        import html
        from concurrent.futures import CancelledError
        from ruinedfooocus.features.gallery.index_store import (
            MediaMetadataError,
            index_connection,
            incremental_scan,
        )

        try:
            self.base_path = self.path_manager.get_abspath(self.path_manager.paths["path_outputs"])
            # Keep the first configured spelling/order, while treating aliases
            # to one directory as one scan root.
            scan_folders = {}
            for folder in [self.base_path] + settings.default_settings.get("archive_folders", []):
                scan_folders.setdefault(Path(folder).resolve(), Path(folder))

            def metadata(path):
                try:
                    return (
                        get_png_metadata(str(path))
                        if path.suffix.lower() == ".png"
                        else get_media_metadata(path)
                    )
                except (ValueError, OSError, Image.DecompressionBombError) as error:
                    # Preserve filesystem/resource failures; only decoder rejection
                    # of a stable individual file is safe to omit from this index.
                    if isinstance(error, OSError) and error.errno in errno.errorcode:
                        raise
                    raise MediaMetadataError(f"{type(error).__name__}: {error}") from error

            with index_connection(self) as connection:
                result = incremental_scan(
                    connection,
                    scan_folders.values(),
                    metadata,
                    cancelled=cancelled,
                    progress=progress,
                    publish=publish,
                    only_if_uninitialized=only_if_uninitialized,
                )
            if result is None:
                return None
            self.last_scan = result

            # Markdown status: resolve "..\" segments and keep Windows backslashes literal.
            folders = ", ".join(f"`{folder.resolve()}`" for folder in scan_folders.values())
            from ruinedfooocus.core.translation_manager import localize

            status = localize(
                "Found {count} images in {folders} and subfolders",
                locale,
                count=result.count,
                folders=folders,
            )
            if result.skipped:
                status += "\n\n" + localize(
                    "Skipped {count} unreadable files: {files}",
                    locale,
                    count=result.skipped,
                    files=html.escape("; ".join(result.warnings)),
                )
            return status
        except CancelledError:
            raise
        except Exception as error:
            from ruinedfooocus.core.translation_manager import localize

            return localize("Error updating folder: {error}", locale, error=error)

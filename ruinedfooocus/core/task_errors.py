"""Actionable descriptions for task failures."""


def describe_error(error):
    detail = f"{type(error).__name__}: {error}"
    lowered = detail.lower()
    if any(
        message in lowered
        for message in (
            "unspecified launch failure",
            "hiperrorlaunchfailure",
            "hiperrorillegaladdress",
            "illegal memory access was encountered",
            "device-side assert triggered",
            "dxgi_error_device_removed",
            "dxgi_error_device_hung",
        )
    ):
        return f"The GPU device failed. Restart RuinedFooocus before generating again. ({detail})"
    if "timeout" in lowered or "timed out" in lowered:
        return f"The operation timed out. Check the connection and retry. Incomplete Hub downloads are kept for resuming. ({detail})"
    if "out of memory" in lowered:
        return f"There was not enough memory. Reduce the model/context size or generation settings, then retry. ({detail})"
    if "could not detect model type" in lowered:
        return (
            "This file is not a supported image checkpoint: its model type was not recognized. "
            "LoRAs, VAEs, text encoders and ControlNets belong in their own model folders. "
            f"({detail.splitlines()[0]})"
        )
    # Truncated or damaged model files (partial downloads, interrupted copies).
    if any(
        message in lowered
        for message in (
            "metadataincompletebuffer",
            "headertoolarge",
            "invalidheader",
            "incomplete metadata",
            "error while deserializing header",
            "failed finding central directory",
            "unexpected end of file",
            "file not fully covered",
        )
    ):
        return (
            "A model file is incomplete or damaged, often from an interrupted download or copy. "
            f"Download it again, or check it with Models > Model > Checkpoint tools. ({detail})"
        )
    if isinstance(error, ValueError):
        return str(error)
    return f"The operation failed. You can retry or choose another model. ({detail})"

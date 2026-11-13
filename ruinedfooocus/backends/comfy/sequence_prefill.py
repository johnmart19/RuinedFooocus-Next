"""Bound Qwen3.5 recurrent prefill temporaries without shortening its context."""

from functools import wraps


def windowed_delta_rule(native, window_size=512):
    """Carry native recurrent state between aligned windows on every device.

    Alignment matters: only the final window may contain native padding. The
    upstream implementation still owns normalization, precision and all math.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive")

    @wraps(native)
    def execute(
        query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False
    ):
        window = max(chunk_size, window_size // chunk_size * chunk_size)
        length = query.shape[1]
        if length <= window:
            return native(
                query,
                key,
                value,
                g,
                beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        import torch

        outputs = []
        state = initial_state
        for start in range(0, length, window):
            tensors = [tensor[:, start : start + window] for tensor in (query, key, value, g, beta)]
            output, state = native(
                *tensors, chunk_size=chunk_size, initial_state=state, output_final_state=True
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1), state if output_final_state else None

    execute._ruinedfooocus_windowed = True
    return execute


def install_qwen35_prefill():
    from comfy.text_encoders import qwen35

    native = qwen35.torch_chunk_gated_delta_rule
    if not getattr(native, "_ruinedfooocus_windowed", False):
        qwen35.torch_chunk_gated_delta_rule = windowed_delta_rule(native)

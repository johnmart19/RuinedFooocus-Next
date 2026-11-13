"""Terminal reporting for automatic accelerator fallback."""

import logging


def warn_cpu_fallback(feature, reason):
    logging.warning("GPU acceleration for %s is unavailable: %s. Using CPU fallback.", feature, reason)

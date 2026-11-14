"""Show the app's own error messages to the person who triggered an event.

Callbacks raise ValueError, PermissionError or FileNotFoundError with text written
for the user ("Select an image in Library first."). Gradio shows only a generic
"Error" for exceptions other than gr.Error, so those three become gr.Error with
the same text. Other exceptions are unexpected and keep Gradio's generic message.
"""

import functools
import inspect
from contextlib import aclosing

import gradio as gr

USER_ERRORS = (ValueError, PermissionError, FileNotFoundError)


def _user_error(error):
    return gr.Error(str(error) or "This action could not be completed.")


def as_user_errors(fn):
    """Wrap one callback, keeping its kind (plain, generator, async) and signature."""
    if getattr(fn, "_rf_user_errors", False):
        return fn
    if inspect.isasyncgenfunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                async with aclosing(fn(*args, **kwargs)) as values:
                    async for item in values:
                        yield item
            except USER_ERRORS as error:
                raise _user_error(error) from error

    elif inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                yield from fn(*args, **kwargs)
            except USER_ERRORS as error:
                raise _user_error(error) from error

    elif inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except USER_ERRORS as error:
                raise _user_error(error) from error

    else:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except USER_ERRORS as error:
                raise _user_error(error) from error

    wrapper._rf_user_errors = True
    return wrapper


def apply(*apps):
    """Wrap every registered event callback of the given Blocks once."""
    for app in apps:
        for block_fn in getattr(app, "fns", {}).values():
            if block_fn.fn is not None:
                block_fn.fn = as_user_errors(block_fn.fn)

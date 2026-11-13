"""Shared visibility update for the existing Create seed control."""

import gradio as gr


def seed_visibility(random_seed):
    return gr.update(visible="hidden" if random_seed else True)

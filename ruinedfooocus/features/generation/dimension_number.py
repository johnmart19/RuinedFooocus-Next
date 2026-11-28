"""Native size controls keep bounds validation without logging input mistakes as crashes."""

import gradio as gr


class DimensionNumber(gr.Number):
    is_template = True

    def preprocess(self, payload):
        try:
            return super().preprocess(payload)
        except gr.Error as error:
            error.print_exception = False
            raise

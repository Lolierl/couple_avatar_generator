import gradio as gr
import numpy as np

def process(image):
    print("process triggered")
    if image is None:
        return []
    return [image]

with gr.Blocks() as demo:
    input_image = gr.Image(source="upload", type="pil")
    run_button = gr.Button("Run")
    gallery = gr.Gallery()

    run_button.click(process, inputs=[input_image], outputs=[gallery])

demo.launch()
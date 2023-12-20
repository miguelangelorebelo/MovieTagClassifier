import gradio as gr

from configs.model_config import AVAILABLE_MODELS, DEFAULT_MODEL
from .slm import classify_text
from utils import CustomFlagger

demo = gr.Interface(fn=classify_text,
                    inputs=["text", gr.Dropdown(AVAILABLE_MODELS, label="model", value=DEFAULT_MODEL)],
                    outputs=[gr.Textbox(label="Tag"), gr.Number(label="Elapsed Time")],
                    title="Movie Tag Predictor",
                    flagging_options=['Incorrect'],
                    flagging_callback=CustomFlagger())

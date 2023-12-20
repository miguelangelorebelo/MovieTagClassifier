import gradio as gr
from fastapi import FastAPI

from modules.gradio_frontend import demo
from routers import predict
from utils import CustomLogger

app = FastAPI(
    title="Movie Tag Classifier API",
    description="""SLM classifier demo""",
    version="1.0.0"
)

# Logger
app.logger = CustomLogger.make_logger()

# LLM API route
app.include_router(predict.router)

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/gradio")

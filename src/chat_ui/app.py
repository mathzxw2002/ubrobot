import os
import shutil
import gradio as gr
import modelscope_studio as mgr
import uvicorn
from fastapi import FastAPI

import time
import logging
logging.basicConfig(level=logging.WARNING)

try:  # Package import for tests and `python -m chat_ui.app`.
    from .pipeline import ChatPipeline
except ImportError:  # Script compatibility: `python src/chat_ui/app.py`.
    from pipeline import ChatPipeline

chat_pipeline = None

def gradio_planning_txt_update():
    while True:
        robot_arm_rgb_image, vis_annotated_img = chat_pipeline.get_robot_observation()
        is_manipulate_valid = robot_arm_rgb_image is not None and robot_arm_rgb_image.size > 0
        yield gr.update(value=vis_annotated_img, visible=True), gr.update(value=robot_arm_rgb_image, visible=is_manipulate_valid)
        time.sleep(1)

def create_gradio():
    with gr.Blocks(title="UBRobot ChatUI") as demo:
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
            UBRobot ChatBot
            </div>  
            """
        )
        with gr.Row():
            with gr.Column(scale = 2):
                gr.Markdown("### Robot Control by Instruction")
                user_chatbot = mgr.Chatbot(
                    label = "Chat History 💬",
                    value = [[None, {"text":"您好，请问有什么可以帮到您？您可以在下方的输入框点击麦克风录制音频或直接输入文本与我聊天。"}],],
                    avatar_images=[
                        {"avatar": os.path.abspath("assets/icon/user.png")},
                        {"avatar": os.path.abspath("assets/icon/qwen.png")},
                    ],
                    height= 500,
                    )
                user_input = mgr.MultimodalInput(sources=["microphone"])

            with gr.Column(scale = 1):
                gr.Markdown("### Nav with Instruction")
                nav_img_output = gr.Image(type="pil", height=320, visible=False)
                manipulate_img_output = gr.Image(type="pil", height=320, visible=False)

        # Use State to store user chat history
        user_messages = gr.State([{'role': 'system', 'content': None}])
        user_processing_flag = gr.State(False)
        lifecycle = mgr.Lifecycle()
        stop_button = gr.Button("Stop")

        # Submit
        user_input.submit(chat_pipeline.run_pipeline,
            inputs=[user_input, user_messages],
            outputs=[user_messages]
        )
        user_input.submit(chat_pipeline.yield_results, 
            inputs=[user_input, user_chatbot, user_processing_flag],
            outputs = [user_input, user_chatbot, user_processing_flag]
        )
        stop_button.click(
            chat_pipeline.stop_pipeline,
            inputs=user_processing_flag,
            outputs=user_processing_flag,
        )

        # refresh
        lifecycle.unmount(chat_pipeline.stop_pipeline, 
            inputs = user_processing_flag, 
            outputs = user_processing_flag
        )
        demo.load(gradio_planning_txt_update, inputs=[], outputs=[nav_img_output, manipulate_img_output])
                
    return demo.queue()

if __name__ == "__main__":
    # UBROBOT_CHAT_MEDIA=off skips ASR/TTS init (offline Windows dev mode).
    media_enabled = os.environ.get("UBROBOT_CHAT_MEDIA", "on").strip().lower() != "off"
    chat_pipeline = ChatPipeline(initialize_media=media_enabled)

    shutil.rmtree('./workspaces/results', ignore_errors= True)

    app = FastAPI()
    gradio_app = create_gradio()
    app = gr.mount_gradio_app(app, gradio_app, path='/')

    uvicorn.run(
        app, 
        host = "0.0.0.0",
        port = 7863, 
        log_level = "warning",
        ssl_keyfile="./assets/key.pem",
        ssl_certfile="./assets/cert.pem"
    )

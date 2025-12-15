import gradio as gr
from fastrtc import WebRTC, ReplyOnPause

import numpy as np


def response(audio: tuple[int, np.ndarray]):
    """This function must yield audio frames"""
    ...

    print("===============================================")
    yield audio


with gr.Blocks() as demo:
    gr.HTML(
    """
    <h1 style='text-align: center'>
    Chat (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    with gr.Column():
        with gr.Group():
            audio = WebRTC(
                mode="send-receive",
                modality="audio",
                #autoplay=True,
            )
        audio.stream(fn=ReplyOnPause(response),
                    inputs=[audio], outputs=[audio],
                    time_limit=60)
demo.launch(
    server_name="192.168.18.233",
    server_port=7866,
    share=False,
    inbrowser=True,
    show_error=True,
    ssl_certfile="./ub_cert.pem",
    ssl_keyfile="./ub_key.pem",
    ssl_verify=False
)

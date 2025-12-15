import gradio as gr
from fastrtc import WebRTC, ReplyOnPause
import traceback # Import traceback
import numpy as np


def response(audio: tuple[int, np.ndarray]):
    """This function must yield audio frames"""
    try:
        # Add your actual processing logic here
        print("Processing audio frame...")
        if audio is None:
            print("Received None audio input.")
        else:
            sample_rate, data = audio
            print(f"Sample rate: {sample_rate}, Data shape: {data.shape}")

        print("===============================================")
        yield audio # Yield the input back for testing
    except Exception as e:
        print(f"An ERROR occurred in response function: {e}")
        traceback.print_exc() # Print the full traceback for debugging
        # You might need to yield something to keep the generator alive, 
        # though the error will likely still disrupt the stream.
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
                    inputs=[audio], 
                    outputs=[audio],
                    time_limit=60,
                    concurrency_limit=1,
        )
demo.launch(
    server_name="0.0.0.0",
    server_port=7866,
    share=True,
    inbrowser=True,
    show_error=True,
    #static_files={
    #    "/manifest.json": "./manifest.json"  # 映射manifest.json路径
    #},
    #ssl_certfile="./ub_cert.pem",
    #ssl_keyfile="./ub_key.pem",
    ssl_verify=False
)
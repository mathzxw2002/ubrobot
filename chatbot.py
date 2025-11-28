import gradio as gr
import random
import time

from pathlib import Path

import qwen_vl_utils
import transformers

ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20


#CosmosReason1Infer cosmos_infer

from cosmos_reason1_infer import CosmosReason1Infer

def create_chatbot_interface() -> gr.Blocks:
    """
    Robot UI
    :return: Robot UI Instance
    """
    with gr.Blocks(title="UBRobot ChatUI") as demo:
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            resut_str = cosmos_infer.infer_once("image_path", message)
            bot_message = resut_str
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    return demo


if __name__ == "__main__":
    demo = create_chatbot_interface()
    
    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason1-7B"
    cosmos_infer = CosmosReason1Infer(model_name)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
        show_error=True
    )
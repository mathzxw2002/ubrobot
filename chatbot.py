import gradio as gr
import random
import time

from pathlib import Path

import qwen_vl_utils
import transformers
import json
import requests


ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20

def cosmos_reason1_infer(image_bytes, instruction, url='http://192.168.18.230:5802/eval_cosmos_reason1'):

    instruction = "trun around to the bag side"
    data = {"ins": instruction}
    json_data = json.dumps(data)

    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    print(f"response {response.text}")
    '''http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")'''

    return json.loads(response.text)

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
            image_bytes = None
            resut_str = cosmos_reason1_infer(image_bytes, message)
            
            bot_message = resut_str
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    return demo


if __name__ == "__main__":
    demo = create_chatbot_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
        show_error=True
    )
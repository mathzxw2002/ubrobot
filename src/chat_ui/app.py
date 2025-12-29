import os
import shutil
import gradio as gr
import modelscope_studio as mgr
import uvicorn
from fastapi import FastAPI
import warnings
warnings.filterwarnings("ignore")
import copy
import time

import logging
import os
from pipeline import chat_pipeline

logging.basicConfig(level=logging.WARNING)

os.environ["DASHSCOPE_API_KEY"] = "sk-479fdd23120c4201bff35a107883c7c3"
os.environ["is_half"] = "True"

shutil.rmtree('./workspaces/results', ignore_errors= True)

from ubrobot.robots.ubrobot import Go2Manager

manager = None

def gradio_planning_txt_update():

    #global planning_response, global_nav_instruction_str, http_idx

    #global_nav_instruction_str = ins_str

    # TODO double check
    #image_bytes = copy.deepcopy(manager.rgb_bytes)
    #result_str = cosmos_reason1_infer(image_bytes, global_nav_instruction_str)
    #result_str = ""

    '''chat_history = []
    chat_history.append({"role": "user", "content": global_nav_instruction_str})
    chat_history.append({"role": "assistant", "content": result_str})

    if global_nav_instruction_str is not None:
        while True:
    
            idx2actions = OrderedDict({"0": "STOP", "1": "↑", "2": "←", "3": "→", "5": "↓", })

            planning_response_str = ""
            pil_annotated_img = None
            if planning_response is not None:

                json_output_dict = planning_response

                pixel_goal = json_output_dict.get('pixel_goal', None)
                traj_path = json_output_dict.get('trajectory', None)
                discrete_act = json_output_dict.get('discrete_action', None)

                planning_response_str = str(idx2actions) + "\n" + str(planning_response)

                pil_annotated_img = annotate_image(http_idx, manager.rgb_image, discrete_act, traj_path, pixel_goal, "./")
             
                yield gr.update(value=planning_response_str), gr.update(value=pil_annotated_img), gr.update(value=chat_history)
            time.sleep(1)'''
    #print(ins_str)

    while True:
        #planning_response_str = ""
        pil_annotated_img = None
        
        #json_output_dict = planning_response

        #pixel_goal = json_output_dict.get('pixel_goal', None)
        #traj_path = json_output_dict.get('trajectory', None)
        #discrete_act = json_output_dict.get('discrete_action', None)

        #planning_response_str = str(idx2actions) + "\n" + str(planning_response)

        #pil_annotated_img = annotate_image(http_idx, manager.rgb_image, discrete_act, traj_path, pixel_goal, "./")
        
        yield gr.update(value=pil_annotated_img)
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

                with gr.Row():
                    avatar_name = gr.Dropdown(label = "数字人形象", choices = ["Avatar1 (通义万相)"], value = "Avatar1 (通义万相)")
                    chat_mode = gr.Dropdown(label = "对话模式", choices = ["单轮对话 (一次性回答问题)", "互动对话 (分多次回答问题)"], value = "单轮对话 (一次性回答问题)")
                    chunk_size = gr.Slider(label = "每次处理的句子最短长度", minimum = 0, maximum = 30, value = 10, step = 1) 
                    tts_module = gr.Dropdown(label = "TTS选型", choices = ["CosyVoice"], value = "CosyVoice")
                    avatar_voice = gr.Dropdown(label = "TTS音色", choices = ["longxiaochun (CosyVoice)", "longwan (CosyVoice)", "longcheng (CosyVoice)", "longhua (CosyVoice)"], value="longwan (CosyVoice)")
                    
                user_input = mgr.MultimodalInput(sources=["microphone"])

            with gr.Column(scale = 1):
                #video_stream = gr.Video(label="Video Stream 🎬 (基于Gradio 5测试版，网速不佳可能卡顿)", streaming=True, height = 500, scale = 1) 
                gr.Markdown("### Nav with Instruction")
                nav_img_output = gr.Image(type="pil", height=480,)
                planning_response_txt = gr.Textbox(interactive=False, lines=5)
                ins_msg_bt = gr.Button("nav instruction")

        # Use State to store user chat history
        user_messages = gr.State([{'role': 'system', 'content': None}])
        user_processing_flag = gr.State(False)
        lifecycle = mgr.Lifecycle()

        # loading TTS Voice
        avatar_voice.change(chat_pipeline.load_voice, 
            inputs=[avatar_voice, tts_module],
            outputs=[user_input]
            )
        lifecycle.mount(chat_pipeline.load_voice,
            inputs=[avatar_voice, tts_module],
            outputs=[user_input]
        )

        # Submit
        user_input.submit(chat_pipeline.run_pipeline,
            inputs=[user_input, user_messages, chunk_size, avatar_name, tts_module, chat_mode],
            outputs=[user_messages]
            )
        user_input.submit(chat_pipeline.yield_results, 
            inputs=[user_input, user_chatbot, user_processing_flag],
            outputs = [user_input, user_chatbot, user_processing_flag]
            )

        # refresh
        lifecycle.unmount(chat_pipeline.stop_pipeline, 
            inputs = user_processing_flag, 
            outputs = user_processing_flag
            )
        
        ins_msg_bt.click(gradio_planning_txt_update, inputs=[], outputs=[nav_img_output])

        #with gr.Row():
            #with gr.Column(scale=1, min_width=300):
                #gr.Markdown("### Nav with Instruction")

                #nav_img_output = gr.Image(type="pil", height=480,)
                #planning_response_txt = gr.Textbox(interactive=False, lines=5)
            
            #with gr.Column(scale=2, min_width=500):
                #gr.Markdown("### Robot Control by Instruction")
                #chatbot = gr.Chatbot(type="messages")
                
                #ins_msg = gr.Textbox(lines=1)

                #with gr.Row():
                #    with gr.Column(scale=1):
                #        ins_msg_bt = gr.Button("nav instruction")
                #    with gr.Column(scale=1):
                #        clear = gr.ClearButton([chatbot])
                #        task_reset_bt = gr.Button("nav task reset")
        
    return demo.queue()

if __name__ == "__main__":
    app = FastAPI()
    gradio_app = create_gradio()
    app = gr.mount_gradio_app(app, gradio_app, path='/')

    manager = Go2Manager()
    
    uvicorn.run(
        app, 
        host = "0.0.0.0",
        port = 7863, 
        log_level = "warning",
        ssl_keyfile="./assets/key.pem",
        ssl_certfile="./assets/cert.pem"
    )

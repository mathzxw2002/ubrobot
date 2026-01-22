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
from pipeline import ChatPipeline

logging.basicConfig(level=logging.WARNING)

os.environ["DASHSCOPE_API_KEY"] = "sk-479fdd23120c4201bff35a107883c7c3"
#os.environ["is_half"] = "True"

shutil.rmtree('./workspaces/results', ignore_errors= True)

from ubrobot.robots.ubrobot import Go2Manager

chat_pipeline = None

def gradio_planning_txt_update():

    '''chat_history = []
    chat_history.append({"role": "user", "content": global_nav_instruction_str})
    chat_history.append({"role": "assistant", "content": result_str})
    '''
    #instruction = "go to the near frontal black bag and stop immediately."
    #manager.set_user_instruction(instruction)
    while True:
        #nav_action, vis_annotated_img = manager.get_next_planning()
        vis_annotated_img =  chat_pipeline.get_nav_vis_image()
        robot_arm_rgb_image = chat_pipeline.get_robot_arm_image_observation()

        #instruction = "Locate objects in current image and return theirs coordinates as json format."
        #robot_arm.grounding_objects_2d(robot_arm_rgb_image, instruction)

        yield gr.update(value=vis_annotated_img), gr.update(value=robot_arm_rgb_image)
        time.sleep(1)

def go2_robot_stop():
    # TODO
    print("stopping the robot.")
    #manager.go2_robot_stop()

def go2_robot_standup():
    print("standing up the robot.")
    #manager.go2_robot_standup()

def go2_robot_standdown():
    print("standing down the robot.")
    #manager.go2_robot_standdown()

def go2_robot_move():
    print("unitree go2 moving test...")
    #manager.go2_robot_move()

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
                gr.Markdown("### Nav with Instruction")
                nav_img_output = gr.Image(type="pil", height=320)
                manipulate_img_output = gr.Image(type="pil", height=320)
                #planning_response_txt = gr.Textbox(interactive=False, lines=5)
                ins_msg_bt = gr.Button("nav instruction")
                #stop_bt = gr.Button("STOP!!!")
                #standup_bt = gr.Button("StandUP")
                #standdown_bt = gr.Button("StandDOWN")
                #move_bt =  gr.Button("MOVE TEST")

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
        
        ins_msg_bt.click(gradio_planning_txt_update, inputs=[], outputs=[nav_img_output, manipulate_img_output])
        #stop_bt.click(go2_robot_stop, inputs=[], outputs=[])
        #standup_bt.click(go2_robot_standup, inputs=[], outputs=[])
        #standdown_bt.click(go2_robot_standdown, inputs=[], outputs=[])
        #move_bt.click(go2_robot_move, inputs=[], outputs=[])
                
    return demo.queue()

if __name__ == "__main__":
    chat_pipeline = ChatPipeline()

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

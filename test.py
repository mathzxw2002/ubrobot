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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
import numpy as np
from PIL import Image as PILImage
import threading

latest_frame = None
frame_lock = threading.Lock()

def cosmos_reason1_infer(image_bytes, instruction, url='http://192.168.18.230:5802/eval_cosmos_reason1'):

    instruction = "trun around to the bag side"
    data = {"ins": instruction}
    json_data = json.dumps(data)

    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    print(f"==================================================response {response.text}")
    '''http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")'''

    return response.text


class ROS2VideoSubscriber(Node):
    def __init__(self, topic_name, is_compressed=False):
        super().__init__("gradio_ros2_video_subscriber")
        self.is_compressed = is_compressed
        
        if self.is_compressed:
            self.subscription = self.create_subscription(
                CompressedImage,
                topic_name,
                self.compressed_image_callback,
                10
            )
        else:
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.raw_image_callback,
                10
            )
        self.get_logger().info(f"ROS2 Topic: {topic_name} (Compressed Format: {is_compressed})")

    def raw_image_callback(self, msg):
        global latest_frame
        try:
            width = msg.width
            height = msg.height
            encoding = msg.encoding

            #print("raw image info:", width, height, encoding)
            
            # 2. 将 ROS2 字节数据转换为 NumPy 数组
            # 注意：不同编码的通道数不同（bgr8/rgb8 是 3 通道，mono8 是 1 通道）
            if encoding in ["bgr8", "rgb8"]:
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            elif encoding == "mono8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 1))
            else:
                self.get_logger().warning(f"不支持的编码格式: {encoding}，默认按 bgr8 解析")
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            
            # 3. 转换为 Gradio 支持的格式（Gradio 默认显示 RGB 图像）
            if encoding == "bgr8":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            
            with frame_lock:
                latest_frame = frame.copy()
                #cv2.imwrite("./test.png", latest_frame)

        except Exception as e:
            self.get_logger().error(f"处理原始图像失败: {str(e)}")

def gradio_video_update():
    global latest_frame
    while True:
        print("in gradio_video_update....")
        with frame_lock:
            print("in gradio video update...", latest_frame)
            if latest_frame is not None:
                #cv2.imwrite("./t.png", latest_frame)
                yield PILImage.fromarray(latest_frame)
            else:
                #print("captured video frame is none or empty...")
                yield PILImage.new("RGB", (640, 480), color="gray")
        time.sleep(0.03)


def respond(message, chat_history):
    image_bytes = None
    resut_str = cosmos_reason1_infer(image_bytes, message)
    
    bot_message = resut_str
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    time.sleep(2)
    return "", chat_history


def create_chatbot_interface() -> gr.Blocks:
    """
    Robot UI
    :return: Robot UI Instance
    """
    with gr.Blocks(title="UBRobot ChatUI") as demo:
        
        gr.Markdown("<h1 style='text-align: center;'>UBRobot Management Demo</h1>")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Image/Video Stream")
                video_output = gr.Image(
                    type="pil",
                    height=480,
                )
                #video_output.stream(gradio_video_update, outputs=video_output)
            
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### Robot Control by Instruction")
                chatbot = gr.Chatbot(
                    #avatar_images=["user.png", "bot.png"]  # 可选：设置头像
                )
                msg = gr.Textbox(lines=1)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot])

        #video_output.stream(gradio_video_update, outputs=video_output)

        test_bt = gr.Button("test")
        test_bt.click(gradio_video_update, inputs=None, outputs=video_output)
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    return demo


if __name__ == "__main__":
    demo = create_chatbot_interface()

    ROS2_VIDEO_TOPIC = "/camera/color/image_raw"
    IS_COMPRESSED = False

    rclpy.init()
    subscriber_node = ROS2VideoSubscriber(ROS2_VIDEO_TOPIC, IS_COMPRESSED)
    
    ros2_thread = threading.Thread(target=rclpy.spin, args=(subscriber_node,), daemon=True)
    ros2_thread.start()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
        show_error=True
    )

    subscriber_node.destroy_node()
    rclpy.shutdown()
    ros2_thread.join()

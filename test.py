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
            # 1. 解析 ROS2 Image 消息（width, height, data）
            width = msg.width
            height = msg.height
            encoding = msg.encoding  # 图像编码（如 bgr8, rgb8）
            
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
            
            # 4. 线程安全更新全局帧
            with frame_lock:
                latest_frame = frame.copy()

        except Exception as e:
            self.get_logger().error(f"处理原始图像失败: {str(e)}")

    def compressed_image_callback(self, msg):
        """处理 sensor_msgs/CompressedImage 压缩图像消息"""
        global latest_frame
        try:
            # 1. 解析压缩图像数据（JPEG/PNG 格式）
            compressed_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            # 2. 用 OpenCV 解码压缩数据
            frame = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)  # 解码为 BGR 格式
            
            # 3. 转换为 Gradio 支持的 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 4. 线程安全更新全局帧
            with frame_lock:
                latest_frame = frame.copy()

        except Exception as e:
            self.get_logger().error(f"处理压缩图像失败: {str(e)}")

def gradio_video_update():
    """Gradio 实时更新函数（每秒返回最新帧）"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                # 转换为 PIL Image（Gradio Image 组件支持 NumPy 数组/PIL 图像）
                yield PILImage.fromarray(latest_frame)
            else:
                # 无帧时返回占位图
                yield PILImage.new("RGB", (640, 480), color="gray")
        time.sleep(0.03)  # 约 30 FPS（根据 Topic 发布频率调整）


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
                    label="图片",
                    height=400,
                    #elem_classes="image-block"  # 绑定自定义CSS类
                )
                video_output.stream(gradio_video_update, outputs=video_output, interval=33)
            
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### Robot Control by Instruction")
                chatbot = gr.Chatbot(
                    #avatar_images=["user.png", "bot.png"]  # 可选：设置头像
                )
                msg = gr.Textbox(lines=1)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot])

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    return demo


if __name__ == "__main__":
    demo = create_chatbot_interface()

    '''ROS2_VIDEO_TOPIC = "/camera/image_raw"  # 替换为你的 ROS2 视频 Topic
    IS_COMPRESSED = False  # 是否为压缩图像（如 /camera/image_raw/compressed 则设为 True）
    # --------------------------------------------------------------

    # 1. 初始化 ROS2
    rclpy.init()
    subscriber_node = ROS2VideoSubscriber(ROS2_VIDEO_TOPIC, IS_COMPRESSED)
    
    # 2. 启动 ROS2 订阅线程（避免阻塞 Gradio）
    ros2_thread = threading.Thread(target=rclpy.spin, args=(subscriber_node,), daemon=True)
    ros2_thread.start()'''


    
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
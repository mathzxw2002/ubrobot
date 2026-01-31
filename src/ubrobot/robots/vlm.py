from openai import OpenAI
from threading import Thread
import json
import time
import requests

import numpy as np
import os, re, cv2
#import random
import io
#import ast
#from io import BytesIO
#from PIL import ImageColor
#import xml.etree.ElementTree as ET

from PIL import Image as PIL_Image

class RobotVLM:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", url = "http://192.168.18.230:5802/eval_cosmos_reason1"):
        self.url = url
        self.client = OpenAI(
            api_key="sk-479fdd23120c4201bff35a107883c7c3",
            base_url=base_url,
        )            
    
    def local_http_service(self, color_image_np, depth_image_np, camera_intrinsic, instruction, url):
        print("calling local deployed http service, url:", url)
        
        print(f"input data type, rgb {color_image_np.dtype}")
        rgb_np = np.ascontiguousarray(color_image_np, dtype=np.uint8)
        
        color_image_pil = PIL_Image.fromarray(rgb_np)
        image_bytes = io.BytesIO()
        color_image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)
        
        depth_image_pil = None
        if depth_image_np is not None:
            print(f"input data type, depth {depth_image_np.dtype}")
            # The depth map as a NumPy array (height, width) of type `np.uint16` (raw depth values in millimeters) and rotation.
            if depth_image_np.dtype != np.uint16:
                raise ValueError(f"Shape of Depth image must be np.uint16，Now is {depth_image_np.dtype}")
            if len(depth_image_np.shape) != 2:
                raise ValueError(f"Shape of Depth image must be (H,W)，Now is {depth_image_np.shape}")
            depth_uint16 = np.ascontiguousarray(depth_image_np).astype(np.uint16)
            
            # Mode "I;16" is specific for 16-bit unsigned integer pixels
            depth_image_pil = PIL_Image.fromarray(depth_uint16, mode='I;16')
            depth_bytes = io.BytesIO()
            depth_image_pil.save(depth_bytes, format='PNG')
            depth_bytes.seek(0)

        if depth_image_pil is not None and camera_intrinsic is not None:
            intrinsic_matrix = [
                [camera_intrinsic.fx, 0.0, camera_intrinsic.ppx],
                [0.0, camera_intrinsic.fy, camera_intrinsic.ppy],
                [0.0, 0.0, 1.0]
            ]
            data = {
                "ins": instruction,
                "intrinsic": intrinsic_matrix
            }
            files = {
                'image': ('rgb_image', image_bytes, 'image/jpeg'),
                'depth': ('depth_image', depth_bytes, 'image/png'),
            }
        else:
            data = {
                "ins": instruction,
            }
            files = {
                'image': ('rgb_image', image_bytes, 'image/jpeg')
            }
        json_data = json.dumps(data)
        try:
            response = requests.post(
                url,
                files=files,
                data={'json': json_data},
                timeout=100
            )
            response.raise_for_status()
            print(f"cosmos_reason1_infer response {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"cosmos_reason1_infer request failed: {e}")
            return ""
        return response.text
    
    def infer(self, user_input, user_messages, chat_mode):
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用长度接近的短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”等短句作为回复的开头。'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages
        )
        print(completion)
        chat_response = completion.choices[0].message.content
        user_messages.append({'role': 'assistant', 'content': chat_response})

        if len(user_messages) > 10:
            user_messages.pop(0)
  
        print(f'[Qwen API] {chat_response}')
        return chat_response, user_messages

    def infer_stream(self, user_input, user_messages, llm_queue, chunk_size, chat_mode):
        print(f"[LLM] User input: {user_input}")
        time_cost = []
        start_time = time.time()
        # prompt 
        if len(user_messages) == 1:
            if chat_mode == "单轮对话 (一次性回答问题)":
                user_messages[0]['content'] = '你负责为一个语音聊天系统生成对话文本输出，使用短句，确保语气情感丰富、友好，并且响应迅速以保持用户的参与感。请你以“好的”、“没问题”、“明白了”、“当然可以”等短句作为回复的开头。'
            else:
                with open('src/prompt.txt', 'r') as f:
                    user_messages[0]['content'] = f.read()
        print(f"[LLM] user_messages: {user_messages}")
        user_messages.append({'role': 'user', 'content': user_input})
        completion = self.client.chat.completions.create(
            model="qwen-turbo",
            messages=user_messages,
            stream=True
        )
        
        chat_response = ""
        buffer = ""
        sentence_buffer = ""
        sentence_split_pattern = re.compile(r'(?<=[,;.!?，；：。:！？》、”])')
        fp_flag = True
        print("[LLM] Start LLM streaming...")
        for chunk in completion:
            chat_response_chunk = chunk.choices[0].delta.content
            chat_response += chat_response_chunk
            buffer += chat_response_chunk

            sentences = sentence_split_pattern.split(buffer)
            
            if not sentences:
                continue
            
            for i in range(len(sentences) - 1):
                sentence = sentences[i].strip()
                sentence_buffer += sentence

                if fp_flag or len(sentence_buffer) >= chunk_size:
                    llm_queue.put(sentence_buffer)
                    time_cost.append(round(time.time()-start_time, 2))
                    start_time = time.time()
                    print(f"[LLM] Put into queue: {sentence_buffer}")
                    sentence_buffer = ""
                    fp_flag = False
            
            buffer = sentences[-1].strip()

        sentence_buffer += buffer
        if sentence_buffer:
            llm_queue.put(sentence_buffer)
            print(f"[LLM] Put into queue: {sentence_buffer}")

        llm_queue.put(None)
        
        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)
        
        print(f"[LLM] Response: {chat_response}\n")
        
        return chat_response, user_messages

    '''def infer_cosmos_reason(self, user_input, user_messages, llm_queue, image_pil, url='http://192.168.18.230:5802/eval_reasoning_vqa_cosmos'):
        """发送图像和指令到HTTP服务，获取推理结果"""
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        user_messages.append({'role': 'user', 'content': user_input})
        print(user_messages)

        instruction = user_input
        data = {"ins": instruction}
        json_data = json.dumps(data)

        files = {
            'image': ('rgb_image', image_bytes, 'image/jpeg'),
        }
        try:
            response = requests.post(
                url,
                files=files,
                data={'json': json_data},
                timeout=120
            )
            response.raise_for_status()
            print(f"cosmos_reason1_infer response {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"cosmos_reason1_infer request failed: {e}")
            return ""

        chat_response = response.text
        if chat_response:
            llm_queue.put(chat_response)
            print(f"[LLM] Put into queue: {chat_response}")

        llm_queue.put(None) 
        user_messages.append({'role': 'assistant', 'content': chat_response})
        if len(user_messages) > 10:
            user_messages.pop(0)

        return chat_response, user_messages'''
    
    def get_vla_rgbd_prompt(self, target_object: str) -> str:
        """
        Generates a prompt for Cosmos Reason 2 that outputs a 2D verification 
        center before the full 3D RGB-D trajectory.
        """
        return (
            f"Analyze the provided video stream from the RealSense camera. Your task is to "
            f"plan a pick-and-place trajectory for the {target_object}.\n\n"
            "**Step 1: Detection Check**\n"
            f"- First, identify the 2D pixel center [u, v] of the {target_object}. "
            "This will be used to verify the detection on the visual feed.\n\n"
            "**Step 2: RGB-D Trajectory Planning**\n"
            "- Generate waypoints in [u, v, d] space (u/v = pixels, d = depth in meters).\n"
            "1. Approach: 5cm above the object center (d - 0.05) | Gripper: 80mm.\n"
            "2. Grasp: At the object center depth (d) | Gripper: 20mm.\n"
            "3. Lift: 15cm toward the camera (d - 0.15) | Gripper: 20mm.\n\n"
            "**Output Format:**\n"
            "Provide reasoning in <think> tags, then a JSON object in <answer> tags:\n"
            "{\n"
            f"  \"target\": \"{target_object}\",\n"
            "  \"bbox_2d\": [x1, y1, x2, y2],\n"
            "  \"trajectory\": [\n"
            "    {\"point\": [u, v, d], \"action\": \"string\", \"gripper_width\": int}\n"
            "  ]\n"
            "}"
        )
    
    def reasoning_vlm_infer(self, image_np, depth_np, intrinc, instruction, url='http://192.168.18.230:5802/eval_reasoning_vqa_cosmos'):
        """发送图像和指令到HTTP服务，获取推理结果"""
        print("=================================================== infer_cosmos_reason")

        #instruction = "Identify the carrot and provide a 3D trajectory for the gripper (which is at the bottom right of the image) to grasp the carrot. Output the trajectory as a JSON list of waypoints with x, y, z, and gripper_width. Format: <answer>your JSON</answer>"

        instruction = self.get_vla_rgbd_prompt("bottle")
        response_str = self.local_http_service(image_np, None, None, instruction, url)
        return response_str

    def vlm_infer_vqa(self, image_np, instruction, url='http://192.168.18.230:5802/eval_reasoning_vqa'):
        #print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(image_np, None, None, instruction, url)
        return response_str
    
    def draw_on_image(self, image, points=None, boxes=None, trajectories=None, output_path=None):
        """
        Draw points, bounding boxes, and trajectories on an image

        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x, y), ...] where x,y are relative (0~1000)
            boxes: List of boxes in format [[x1, y1, x2, y2], ...] where coords are relative (0~1000)
            trajectories: List of trajectories in format [[(x, y), (x, y), ...], ...]
                        or [[(x, y, d), ...], ...] where x,y are relative (0~1000)
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            if image is None:
                raise FileNotFoundError(f"Unable to load image: {image}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                """Convert relative (0~1000) to absolute pixel coords, clamped to image bounds."""
                x = int(round((x_rel / 1000.0) * w))
                y = int(round((y_rel / 1000.0) * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            # Draw points
            if points:
                for point in points:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1r, y1r, x2r, y2r = box
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if not trajectory or len(trajectory) < 2:
                        continue

                    # Convert all trajectory points to absolute pixels
                    abs_pts = []
                    for p in trajectory:
                        # support (x,y) or (x,y,d)
                        x_rel, y_rel = p[0], p[1]
                        abs_pts.append(rel_to_abs(x_rel, y_rel))

                    # Connect trajectory points with lines
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 0, 255), 2)  # Blue line

                    # Draw a larger point at the trajectory end
                    start_x, start_y = abs_pts[0]
                    cv2.circle(image, (start_x, start_y), 7, (0, 255, 0), -1)  # Red start point

                    # Draw a larger point at the trajectory end
                    end_x, end_y = abs_pts[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)  # Blue end point

            # Save the result
            #output_path = "./test.png"
            #cv2.imwrite(output_path, image)
            #print(f"Annotated image saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def vlm_infer_traj(self, rgb_image_np, depth_image_np, intrin, instruction, url='http://192.168.18.230:5802/eval_reasoning_traj'):
        #print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(rgb_image_np, depth_image_np, intrin, instruction, url)
        return response_str

    def vlm_infer_grounding(self, image_np, instruction, url='http://192.168.18.230:5802/eval_reasoning_grounding'):
        #print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(image_np, None, None, instruction, url)
        boxes = self.decode_json_points(response_str)
        self.draw_on_image(image_np, None, [boxes], None, None)
        return response_str
    
    def decode_json_points(self, text: str):
        """Parse coordinate points from text format"""
        try:
            print("=======================", text)
            answer_str = json.loads(text).get("answer", "{}")
            print(answer_str)
            bbox_2d = json.loads(answer_str).get("bbox_2d", [])

            print(bbox_2d)

            bbox_2d_int = [int(num) for num in bbox_2d]
            return bbox_2d_int
        except Exception as e:
            print(f"Error: {e}")
            return []

    def grounding_2d_bbox(self, image_bytes):
        json_output = None

        instruction_prompt = 'locate every instance, and report bbox coordinates in JSON format.'
        data = {"ins": instruction_prompt}

        json_data = json.dumps(data)

        files = {
            'image': ('rgb_image', image_bytes, 'image/jpeg'),
        }
        try:
            response = requests.post(
                self.url,
                files=files,
                data={'json': json_data},
                timeout=100
            )
            response.raise_for_status()
            print(f"cosmos_reason1_infer response {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"cosmos_reason1_infer request failed: {e}")
            return ""
        json_output = response.text

        # decode json_output to object list
        return json_output
    

if __name__ == "__main__":
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")

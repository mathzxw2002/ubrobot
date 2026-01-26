from openai import OpenAI
from threading import Thread
import re
import queue
import os
import json
import time
import requests

import random
import io
import ast
from io import BytesIO
from PIL import ImageColor
import xml.etree.ElementTree as ET

from PIL import Image as PIL_Image

class RobotVLM:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1", url = "http://192.168.18.230:5802/eval_cosmos_reason1"):
        self.url = url
        self.client = OpenAI(
            api_key="sk-479fdd23120c4201bff35a107883c7c3",
            base_url=base_url,
        )
    
    def local_http_service(self, image_pil, instruction, url):
        print("calling local deployed http service, url:", url)
        
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

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

    def infer_cosmos_reason(self, user_input, user_messages, llm_queue, image_pil, url='http://192.168.18.230:5802/eval_cosmos_reason1'):
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

        return chat_response, user_messages
    
    def reasoning_vlm_infer(self, image_pil, instruction):
        return self._cosmos_reason1_infer(image_pil, instruction)

    def _cosmos_reason1_infer(self, image_pil, instruction):
        """发送图像和指令到HTTP服务，获取推理结果"""
        print("=================================================== infer_cosmos_reason")
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        data = {"ins": instruction}
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

        return response.text

    def vlm_infer_vqa(self, image_pil, instruction, url='http://192.168.18.230:5802/eval_reasoning_vqa'):
        print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(image_pil, instruction, url)
        return response_str
    
    def vlm_infer_traj(self, image_pil, instruction, url='http://192.168.18.230:5802/eval_reasoning_traj'):
        print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(image_pil, instruction, url)
        return response_str

    def vlm_infer_grounding(self, image_pil, instruction, url='http://192.168.18.230:5802/eval_reasoning_grounding'):
        print("eval robobrain 2.5 ...")
        response_str = self.local_http_service(image_pil, instruction, url)
        return response_str
    
    def decode_json_points(self, text: str):
        """Parse coordinate points from text format"""
        try:
            # 清理markdown标记
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            
            # 解析JSON
            data = json.loads(text)
            points = []
            labels = []
            
            for item in data:
                if "point_2d" in item:
                    x, y = item["point_2d"]
                    points.append([x, y])
                    
                    # 获取label，如果没有则使用默认值
                    label = item.get("label", f"point_{len(points)}")
                    labels.append(label)
            
            return points, labels
        except Exception as e:
            print(f"Error: {e}")
            return [], []
        

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

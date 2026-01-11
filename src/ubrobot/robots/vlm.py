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
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

class RobotVLM:
    def __init__(self, url = "http://192.168.18.230:5802/eval_cosmos_reason1"):
        self.url = url
       
    
    def reasoning_vlm_infer(self, image_bytes, instruction):
        # TODO test
        self.grounding_2d_bbox(image_bytes)
        return self._cosmos_reason1_infer(image_bytes, instruction)

    def _cosmos_reason1_infer(self, image_bytes, instruction):
        """发送图像和指令到HTTP服务，获取推理结果"""
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

from openai import OpenAI
from threading import Thread
import re
import queue
import os
import json
import time
import requests


class RobotVLM:
    def __init__(self, url = "http://192.168.18.230:5802/eval_cosmos_reason1"):
        self.url = url
       
    
    def reasoning_vlm_infer(self, image_bytes, instruction):
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

if __name__ == "__main__":
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")

from openai import OpenAI
from threading import Thread
import re
import queue
import os
import json
import time


class RobotNav:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        api_key = api_key if api_key else os.getenv("DASHSCOPE_API_KEY")
        self.client = OpenAI(
            #api_key=os.getenv("DASHSCOPE_API_KEY"),
            api_key="sk-479fdd23120c4201bff35a107883c7c3",
            base_url=base_url,
        )
       
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

if __name__ == "__main__":
    start_time = time.time()
    qwen = Qwen()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    qwen.infer_stream("讲一个长点的故事", [{'role': 'system', 'content': None}], None, None, "单轮对话 (一次性回答问题)")
    print(f"Cost {time.time()-start_time} secs")

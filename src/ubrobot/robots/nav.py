from openai import OpenAI
from threading import Thread
import re
import queue
import os
import json
import time
from PIL import ImageDraw, ImageFont
import cv2
from PIL import Image as PIL_Image
import requests

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
    
    def _annotate_image(self, idx, image, llm_output, trajectory, pixel_goal, output_dir):
        """可视化轨迹和像素目标，给图像添加标注"""
        image = PIL_Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        font_size = 20
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        text_content = []
        text_content.append(f"Frame    Id  : {idx}")
        text_content.append(f"Actions      : {llm_output}")
        max_width = 0
        total_height = 0
        for line in text_content:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = 26
            max_width = max(max_width, text_width)
            total_height += text_height

        padding = 10
        box_x, box_y = 10, 10
        box_width = max_width + 2 * padding
        box_height = total_height + 2 * padding

        draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill='black')

        text_color = 'white'
        y_position = box_y + padding

        for line in text_content:
            draw.text((box_x + padding, y_position), line, fill=text_color, font=font)
            y_position += 26
        image = np.array(image)

        # 绘制轨迹可视化图
        if trajectory is not None and len(trajectory) > 0:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            img_height, img_width = image.shape[:2]
            window_size = 200
            window_x = img_width - window_size
            window_y = 0

            # 提取轨迹点
            traj_points = []
            for point in trajectory:
                if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                    traj_points.append([float(point[0]), float(point[1])])

            if len(traj_points) > 0:
                traj_array = np.array(traj_points)
                x_coords = traj_array[:, 0]
                y_coords = traj_array[:, 1]

                # 创建matplotlib图
                fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
                fig.patch.set_alpha(0.6)
                fig.patch.set_facecolor('gray')
                ax.set_facecolor('lightgray')

                # 绘制轨迹
                ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')
                ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
                ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')
                ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')

                ax.set_xlabel('Y (left +)', fontsize=8)
                ax.set_ylabel('X (up +)', fontsize=8)
                ax.invert_xaxis()
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_aspect('equal', adjustable='box')
                ax.legend(fontsize=6, loc='upper right')
                plt.tight_layout(pad=0.3)

                # 转换为numpy数组
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
                plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., 1:4]
                plt.close(fig)

                # 缩放并叠加到原图
                plot_img = cv2.resize(plot_img, (window_size, window_size))
                image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

        # 绘制像素目标点
        if pixel_goal is not None:
            cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)
        image = PIL_Image.fromarray(image).convert('RGB')
        return image
    
    def _dual_sys_eval(self, policy_init, http_idx, image_bytes, depth_bytes, instruction, url='http://192.168.18.230:5801/eval_dual'):
        
        #global frame_data
        data = {"reset": policy_init, "idx": http_idx, "ins": instruction}
        json_data = json.dumps(data)
        
        files = {
            'image': ('rgb_image', image_bytes, 'image/jpeg'),
            'depth': ('depth_image', depth_bytes, 'image/png'),
        }
        
        try:
            response = requests.post(
                url,
                files=files,
                data={'json': json_data},
                timeout=100
            )
            response.raise_for_status()
            print(f"dual_sys_eval response {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"dual_sys_eval request failed: {e}")
            return {}

        return json.loads(response.text)

if __name__ == "__main__":
    start_time = time.time()
    qwen = Qwen()
    print(f"Cost {time.time()-start_time} secs")
    start_time = time.time()
    qwen.infer_stream("讲一个长点的故事", [{'role': 'system', 'content': None}], None, None, "单轮对话 (一次性回答问题)")
    print(f"Cost {time.time()-start_time} secs")

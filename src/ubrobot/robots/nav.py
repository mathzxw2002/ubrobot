import json
import time
from PIL import ImageDraw, ImageFont
import cv2
from PIL import Image as PIL_Image
import requests
import io

from collections import OrderedDict

import math
import numpy as np

from enum import Enum

class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2

class RobotAction:
    def __init__(self):
        self.current_control_mode = ControlMode.MPC_Mode
        self.trajs_in_world = None
        self.actions = None
        self.homo_goal = None
        self.odom = None
        self.homo_odom = None

class RobotNav:
    def __init__(self, api_key = None, base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.idx2actions = OrderedDict({"0": "STOP", "1": "↑", "2": "←", "3": "→", "5": "↓", })
        
           
    def _annotate_image(self, idx, rgb_bytes, llm_output, trajectory, pixel_goal, odom):
        """可视化轨迹和像素目标，给图像添加标注"""
        #image = PIL_Image.fromarray(image)
        rgb_bytes.seek(0)
        # 从字节流中读取图像，返回 PIL.Image.Image 对象
        image = PIL_Image.open(rgb_bytes)

        draw = ImageDraw.Draw(image)
        font_size = 20
        font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
        text_content = []
        text_content.append(f"Frame    Id  : {idx}")

        if llm_output is not None:
            text_content.append(f"Actions      : {llm_output}")

            action_list = []
            for num in llm_output:
                num_str = str(num)
                action = self.idx2actions.get(num_str, "-")
                action_list.append(action)
            text_content.append(f"Actions      : {action_list}")

        shot_odom = []
        for i in odom:
            shot_odom.append(f"{i:.2f}")
        text_content.append(f"Odom      : {shot_odom}")

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
    
    def incremental_change_goal(self, actions, homo_odom):

        homo_goal = homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                # 前进
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                # 左转15度
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                # 右转15度
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        return homo_goal
    
    def convert_policy_res_to_action(self, response, odom):
        # compute homo_odom by odom
        # 计算齐次变换矩阵
        yaw = odom[2]
        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        homo_odom = np.eye(4)
        homo_odom[:2, :2] = R0
        homo_odom[:2, 3] = odom[:2]

        act = RobotAction()
        act.odom = odom
        act.homo_odom = homo_odom

        if 'trajectory' in response:
            trajectory = response['trajectory']

            trajs_in_world = []
            traj_len = np.linalg.norm(trajectory[-1][:2])
            print(f"traj len {traj_len}")

            # 转换轨迹到世界坐标系
            for i, traj in enumerate(trajectory):
                if i < 3:
                    continue
                x_, y_, yaw_ = odom[0], odom[1], odom[2]
                w_T_b = np.array(
                    [
                        [np.cos(yaw_), -np.sin(yaw_), 0, x_],
                        [np.sin(yaw_), np.cos(yaw_), 0, y_],
                        [0.0, 0.0, 1.0, 0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                w_P = (w_T_b @ (np.array([traj[0], traj[1], 0.0, 1.0])).T)[:2]
                trajs_in_world.append(w_P)
            trajs_in_world = np.array(trajs_in_world)
            print(f"{time.time()} update traj")
            act.trajs_in_world = trajs_in_world

            print("====================check traj. in world.", act.trajs_in_world)
            
            act.current_control_mode = ControlMode.MPC_Mode
        elif 'discrete_action' in response:
            # 离散动作：切换到PID模式
            actions = response['discrete_action']
            act.actions = actions
            if actions != [5] and actions != [9]:
                act.homo_goal = self.incremental_change_goal(actions, homo_odom)
                act.current_control_mode = ControlMode.PID_Mode
        return act
    
    def _dual_sys_eval(self, policy_init, http_idx, rgb_image, depth, instruction, odom, url='http://192.168.18.230:5801/eval_dual'):
        
        rgb_image_pil = PIL_Image.fromarray(rgb_image)
        depth_pil = PIL_Image.fromarray(depth)
        
        image_bytes = io.BytesIO()
        rgb_image_pil.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        depth_bytes = io.BytesIO()
        depth_pil.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

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

        nav_result = json.loads(response.text)
        nav_action = self.convert_policy_res_to_action(nav_result, odom)

        pixel_goal = nav_result.get('pixel_goal', None)
        traj_path = nav_result.get('trajectory', None)
        discrete_act = nav_result.get('discrete_action', None)
        vis_annotated_img = self._annotate_image(http_idx, image_bytes, discrete_act, traj_path, pixel_goal, odom)

        return nav_action, vis_annotated_img
        

if __name__ == "__main__":
    start_time = time.time()
    print(f"Cost {time.time()-start_time} secs")

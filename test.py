import gradio as gr
import random
import time

from pathlib import Path


from collections import OrderedDict
import qwen_vl_utils
import transformers
import json
import requests

import copy
import io
import math
from collections import deque
from enum import Enum

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
import requests

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PIL_Image
from PIL import ImageDraw, ImageFont
# user-specific
from controllers import Mpc_controller, PID_controller
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from thread_utils import ReadWriteLock

from lekiwi.lekiwi_base import LeKiwi
from lekiwi.config_lekiwi_base import LeKiwiConfig

from sensor_msgs.msg import Image, CompressedImage
import cv2
import threading

ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20

frame_lock = threading.Lock()

frame_data = {}

class ControlMode(Enum):
    PID_Mode = 1
    MPC_Mode = 2


# global variable
policy_init = True
mpc = None
pid = PID_controller(Kp_trans=2.0, Kd_trans=0.0, Kp_yaw=1.5, Kd_yaw=0.0, max_v=0.6, max_w=0.5)
http_idx = -1
first_running_time = 0.0
last_s2_step = -1
manager = None
current_control_mode = ControlMode.MPC_Mode
trajs_in_world = None

desired_v, desired_w = 0.0, 0.0
rgb_depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
mpc_rw_lock = ReadWriteLock()


planning_response = None

global_nav_instruction_str = None

planning_thread_instance = None

# visualize tracjectory and pixel goal image
def annotate_image(idx, image, llm_output, trajectory, pixel_goal, output_dir):
    image = PIL_Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    text_content = []
    text_content.append(f"Frame    Id  : {idx}")
    text_content.append(f"Actions      : {llm_output}" )
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
        bbox = draw.textbbox((0, 0), line, font=font)
        text_height = 26
        y_position += text_height
    image = np.array(image)

    # Draw trajectory visualization in the top-right corner using matplotlib
    if trajectory is not None and len(trajectory) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        img_height, img_width = image.shape[:2]

        # Window parameters
        window_size = 200  # Window size in pixels
        window_margin = 0  # Margin from edge
        window_x = img_width - window_size - window_margin
        window_y = window_margin

        # Extract trajectory points
        traj_points = []
        for point in trajectory:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                traj_points.append([float(point[0]), float(point[1])])

        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords = traj_array[:, 0]
            y_coords = traj_array[:, 1]

            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)  # Semi-transparent background
            fig.patch.set_facecolor('gray')
            ax.set_facecolor('lightgray')

            # Plot trajectory
            # Coordinate system: x-axis points up, y-axis points left
            # Origin at bottom center
            ax.plot(y_coords, x_coords, 'b-', linewidth=2, label='Trajectory')

            # Mark start point (green) and end point (red)
            ax.plot(y_coords[0], x_coords[0], 'go', markersize=6, label='Start')
            ax.plot(y_coords[-1], x_coords[-1], 'ro', markersize=6, label='End')

            # Mark origin
            ax.plot(0, 0, 'w+', markersize=10, markeredgewidth=2, label='Origin')

            # Set axis labels
            ax.set_xlabel('Y (left +)', fontsize=8)
            ax.set_ylabel('X (up +)', fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)

            # Set equal aspect ratio
            ax.set_aspect('equal', adjustable='box')

            # Add legend
            ax.legend(fontsize=6, loc='upper right')

            # Adjust layout
            plt.tight_layout(pad=0.3)

            # Convert matplotlib figure to numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            plot_img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

            #print("check value ...")
            #print(fig.canvas.get_width_height()[::-1])
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[...,1:4]
            plt.close(fig)

            # Resize plot to fit window
            plot_img = cv2.resize(plot_img, (window_size, window_size))

            # Overlay plot on image
            image[window_y:window_y+window_size, window_x:window_x+window_size] = plot_img

    if pixel_goal is not None:
        cv2.circle(image, (pixel_goal[1], pixel_goal[0]), 5, (255, 0, 0), -1)
    image = PIL_Image.fromarray(image).convert('RGB')
    #print("saving image.....................")
    #print(f'++++++++++++++++++++++++++++++++++++++++{output_dir}/rgb_{idx}_annotated.png')
    #image.save(f'{output_dir}/rgb_{idx}_annotated.png')
    # to numpy array

    #cv2.imshow("vis_dul_sys_traj", image)
    return image


def dual_sys_eval(image_bytes, depth_bytes, front_image_bytes, url='http://192.168.18.230:5801/eval_dual'):
    global policy_init, http_idx, first_running_time, global_nav_instruction_str

    #instruction = "Turn around and walk out of this office. Turn towards your slight right at the chair. Move forward to the walkway and go near the red bin. You can see an open door on your right side, go inside the open door. Stop at the computer monitor"
    #instruction = "walk close to office chair, walk away from the package with SANY brand."
    #instruction = "turn around to the office chair side."
    instruction = global_nav_instruction_str
    data = {"reset": policy_init, "idx": http_idx, "ins": instruction}
    json_data = json.dumps(data)

    policy_init = False
    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
        'depth': ('depth_image', depth_bytes, 'image/png'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    print(f"response {response.text}")
    http_idx += 1
    if http_idx == 0:
        first_running_time = time.time()
    print(f"idx: {http_idx} after http {time.time() - start}")

    return json.loads(response.text)

def control_thread():
    global desired_v, desired_w, global_nav_instruction_str


    while True:
        global current_control_mode

        if global_nav_instruction_str is None:
            time.sleep(0.01)
            continue

        print("=============== in control thread...", current_control_mode)
        if current_control_mode == ControlMode.MPC_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            if mpc is not None and manager is not None and odom is not None:
                local_mpc = mpc
                opt_u_controls, opt_x_states = local_mpc.solve(np.array(odom))
                v, w = opt_u_controls[0, 0], opt_u_controls[0, 1]

                desired_v, desired_w = v, w
                manager.move(v, 0.0, w)
        elif current_control_mode == ControlMode.PID_Mode:
            odom_rw_lock.acquire_read()
            odom = manager.odom.copy() if manager.odom else None
            odom_rw_lock.release_read()
            homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
            vel = manager.vel.copy() if manager.vel is not None else None
            homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None

            #print("homo_odom, vel, homo_goal...", homo_odom, vel, homo_goal)
            if homo_odom is not None and vel is not None and homo_goal is not None:
                v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
                if v < 0.0:
                    v = 0.0
                desired_v, desired_w = v, w

                print(v, w)
                manager.move(v, 0.0, w)

        time.sleep(0.1)


def planning_thread():
    global trajs_in_world, global_nav_instruction_str

    while True:
        start_time = time.time()
        DESIRED_TIME = 0.3
        time.sleep(0.05)

        if not manager.new_image_arrived:
            time.sleep(0.01)
            continue

        # wait for new instruction
        if global_nav_instruction_str is None:
            time.sleep(0.01)
            continue
        manager.new_image_arrived = False
        rgb_depth_rw_lock.acquire_read()
        rgb_bytes = copy.deepcopy(manager.rgb_bytes)
        depth_bytes = copy.deepcopy(manager.depth_bytes)
        infer_rgb = copy.deepcopy(manager.rgb_image)
        infer_depth = copy.deepcopy(manager.depth_image)
        rgb_time = manager.rgb_time
        rgb_depth_rw_lock.release_read()
        odom_rw_lock.acquire_read()
        min_diff = 1e10
        # time_diff = 1e10
        odom_infer = None
        for odom in manager.odom_queue:
            diff = abs(odom[0] - rgb_time)
            if diff < min_diff:
                min_diff = diff
                odom_infer = copy.deepcopy(odom[1])
                # time_diff = odom[0] - rgb_time
        # odom_time = manager.odom_timestamp
        odom_rw_lock.release_read()

        if odom_infer is not None and rgb_bytes is not None and depth_bytes is not None:
            global frame_data
            frame_data[http_idx] = {
                'infer_rgb': copy.deepcopy(infer_rgb),
                'infer_depth': copy.deepcopy(infer_depth),
                'infer_odom': copy.deepcopy(odom_infer),
            }
            if len(frame_data) > 100:
                del frame_data[min(frame_data.keys())]
            response = dual_sys_eval(rgb_bytes, depth_bytes, None)

            global planning_response
            planning_response = response

            global current_control_mode
            traj_len = 0.0
            if 'trajectory' in response:
                trajectory = response['trajectory']
                trajs_in_world = []
                odom = odom_infer
                traj_len = np.linalg.norm(trajectory[-1][:2])
                print(f"traj len {traj_len}")
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

                manager.last_trajs_in_world = trajs_in_world
                mpc_rw_lock.acquire_write()
                global mpc
                if mpc is None:
                    mpc = Mpc_controller(np.array(trajs_in_world))
                else:
                    mpc.update_ref_traj(np.array(trajs_in_world))
                manager.request_cnt += 1
                mpc_rw_lock.release_write()
                current_control_mode = ControlMode.MPC_Mode
            elif 'discrete_action' in response:
                actions = response['discrete_action']
                if actions != [5] and actions != [9]:
                    manager.incremental_change_goal(actions)
                    current_control_mode = ControlMode.PID_Mode
                if actions == [0]:
                    global_nav_instruction_str = None
        else:
            print(
                f"skip planning. odom_infer: {odom_infer is not None} rgb_bytes: {rgb_bytes is not None} depth_bytes: {depth_bytes is not None}"
            )
            time.sleep(0.1)

        time.sleep(max(0, DESIRED_TIME - (time.time() - start_time)))


class Go2Manager(Node):
    def __init__(self):
        super().__init__('go2_manager')

        rgb_down_sub = Subscriber(self, Image, "/camera/color/image_raw")
        depth_down_sub = Subscriber(self, Image, "/camera/aligned_depth_to_color/image_raw")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self.syncronizer = ApproximateTimeSynchronizer([rgb_down_sub, depth_down_sub], 1, 0.1)
        self.syncronizer.registerCallback(self.rgb_depth_down_callback)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile)

        # publisher
        self.control_pub = self.create_publisher(Twist, '/cmd_vel_bridge', 5)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.rgb_bytes = None
        self.depth_image = None
        self.depth_bytes = None
        self.rgb_forward_image = None
        self.rgb_forward_bytes = None
        self.new_image_arrived = False
        self.new_vis_image_arrived = False
        self.rgb_time = 0.0

        self.odom = None
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.request_cnt = 0
        self.odom_cnt = 0
        self.odom_queue = deque(maxlen=50)
        self.odom_timestamp = 0.0

        self.last_s2_step = -1
        self.last_trajs_in_world = None
        self.last_all_trajs_in_world = None
        self.homo_odom = None
        self.homo_goal = None
        self.vel = None

        # init lekiwi base robot
        lekiwi_base_config = LeKiwiConfig()
        self.lekiwi_base = LeKiwi(lekiwi_base_config)

        self.lekiwi_base.connect()

    def rgb_forward_callback(self, rgb_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_forward_image = raw_image
        image = PIL_Image.fromarray(self.rgb_forward_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.rgb_forward_bytes = image_bytes
        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def rgb_depth_down_callback(self, rgb_msg, depth_msg):
        raw_image = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')[:, :, :]
        self.rgb_image = raw_image
        image = PIL_Image.fromarray(self.rgb_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        raw_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth = (np.clip(self.depth_image * 10000.0, 0, 65535)).astype(np.uint16)
        depth = PIL_Image.fromarray(depth)
        depth_bytes = io.BytesIO()
        depth.save(depth_bytes, format='PNG')
        depth_bytes.seek(0)

        rgb_depth_rw_lock.acquire_write()
        self.rgb_bytes = image_bytes

        self.rgb_time = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec / 1.0e9
        self.last_rgb_time = self.rgb_time

        self.depth_bytes = depth_bytes
        self.depth_time = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec / 1.0e9
        self.last_depth_time = self.depth_time

        rgb_depth_rw_lock.release_write()

        self.new_vis_image_arrived = True
        self.new_image_arrived = True

    def odom_callback(self, msg):
        self.odom_cnt += 1
        odom_rw_lock.acquire_write()
        zz = msg.pose.pose.orientation.z
        ww = msg.pose.pose.orientation.w
        yaw = math.atan2(2 * zz * ww, 1 - 2 * zz * zz)
        self.odom = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
        self.odom_queue.append((time.time(), copy.deepcopy(self.odom)))
        self.odom_timestamp = time.time()
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        odom_rw_lock.release_write()

        R0 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.vel = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]

        if self.odom_cnt == 1:
            self.homo_goal = self.homo_odom.copy()

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_odom.copy()
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array(
                    [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
                )
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
        self.homo_goal = homo_goal

    def move(self, vx, vy, vyaw):
        request = Twist()
        request.linear.x = vx
        request.linear.y = 0.0
        request.angular.z = vyaw

        self.control_pub.publish(request)

        action = {"x.vel": vx, 
                  "y.vel": 0,
                  "theta.vel": vyaw
                  }
        self.lekiwi_base.send_action(action)


def cosmos_reason1_infer(image_bytes, instruction, url='http://192.168.18.230:5802/eval_cosmos_reason1'):

    data = {"ins": instruction}
    json_data = json.dumps(data)

    files = {
        'image': ('rgb_image', image_bytes, 'image/jpeg'),
    }
    start = time.time()
    response = requests.post(url, files=files, data={'json': json_data}, timeout=100)
    print(f"==================================================response {response.text}")
    return response.text


'''class ROS2VideoSubscriber(Node):
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
                yield PIL_Image.fromarray(latest_frame)
            else:
                #print("captured video frame is none or empty...")
                yield PIL_Image.new("RGB", (640, 480), color="gray")
        time.sleep(0.03)
'''

def gradio_planning_txt_update(ins_str):

    global planning_response, global_nav_instruction_str, http_idx

    global_nav_instruction_str = ins_str

    # TODO double check
    image_bytes = copy.deepcopy(manager.rgb_bytes)
    #result_str = cosmos_reason1_infer(image_bytes, global_nav_instruction_str)
    result_str = ""

    chat_history = []
    chat_history.append({"role": "user", "content": global_nav_instruction_str})
    chat_history.append({"role": "assistant", "content": result_str})

    if global_nav_instruction_str is not None:
        while True:
    
            idx2actions = OrderedDict({"0": "STOP", "1": "↑", "2": "←", "3": "→", "5": "↓", })

            planning_response_str = ""
            pil_annotated_img = None
            if planning_response is not None:

                json_output_dict = planning_response

                pixel_goal = json_output_dict.get('pixel_goal', None)
                traj_path = json_output_dict.get('trajectory', None)
                discrete_act = json_output_dict.get('discrete_action', None)

                planning_response_str = str(idx2actions) + "\n" + str(planning_response)

                pil_annotated_img = annotate_image(http_idx, manager.rgb_image, discrete_act, traj_path, pixel_goal, "./")
             
                yield gr.update(value=planning_response_str), gr.update(value=pil_annotated_img), gr.update(value=chat_history)
            time.sleep(1)


def nav_task_reset():

    global global_nav_instruction_str

    global_nav_instruction_str = None
    #task_completed = False

    # TODO stop the robot
    

def create_chatbot_interface() -> gr.Blocks:
    """
    Robot UI
    :return: Robot UI Instance
    """
    with gr.Blocks(title="UBRobot ChatUI") as demo:
        
        gr.Markdown("<h1 style='text-align: center;'>UBRobot Management Demo</h1>")

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Nav with Instruction")

                nav_img_output = gr.Image(type="pil", height=480,)
                planning_response_txt = gr.Textbox(interactive=False, lines=5)
            
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### Robot Control by Instruction")
                chatbot = gr.Chatbot()
                
                ins_msg = gr.Textbox(lines=1)
                with gr.Row():
                    with gr.Column(scale=1):
                        ins_msg_bt = gr.Button("nav instruction")
                    with gr.Column(scale=1):
                        clear = gr.ClearButton([chatbot])
                        task_reset_bt = gr.Button("nav task reset")

        ins_msg_bt.click(gradio_planning_txt_update, inputs=ins_msg, outputs=[planning_response_txt, nav_img_output, chatbot])
        task_reset_bt.click(nav_task_reset, inputs=None, outputs=None)
    return demo

def run_launch():
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    demo = create_chatbot_interface()

    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)
    control_thread_instance.daemon = True
    planning_thread_instance.daemon = True

    rclpy.init()
    
    print("start robot client...")
    launching_thread = threading.Thread(target=run_launch, daemon=True)
    launching_thread.start()

    manager = Go2Manager()

    executor = SingleThreadedExecutor()
    executor.add_node(manager)

    control_thread_instance.start()
    planning_thread_instance.start()
    
    executor.spin()

    executor.shutdown()
    subscriber_node.destroy_node()
    manager.destroy_node()
    rclpy.shutdown()

      
import argparse
import json
import os
import time
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from cosmos_reason1_infer import CosmosReason1Agent

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
# Flask 启动时加载模型一次
cosmos_agent = CosmosReason1Agent("/media/agi/公共盘/cosmos-reason1/Cosmos-Reason1-7B")

@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time
    start_time = time.time()

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth_image = depth.copy()
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = "Walk to the cardboard box in front."
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t0 = time.time()
    dual_sys_output = {}

    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
    if dual_sys_output.output_pixel is not None:
        json_output['pixel_goal'] = dual_sys_output.output_pixel
        pixel_goal = dual_sys_output.output_pixel
        # 复制原始图像用于绘制
        image_with_point = image.copy()
        # 在目标像素位置绘制红色圆点
        cv2.circle(image_with_point, (pixel_goal[1], pixel_goal[0]), radius=5, color=(255, 0, 0), thickness=-1)
        # 保存带标记点的图像
        image_pil = Image.fromarray(image_with_point)
        image_pil.save(os.path.join(output_dir, f'image_with_point_{idx}.jpg'), 'JPEG', quality=95)
        depth_image.save(os.path.join(output_dir, f'depth_image_{idx}.png'))
        print(f"output_trajectory: {dual_sys_output}")
        image_dir = os.path.join(output_dir, f'image_with_point_{idx}.jpg')   
            # 调用 Cosmos-Reason1 进行视觉推理
        cosmos_result = cosmos_agent.infer_once(
            image_path=image_dir,
            action_seq=dual_sys_output.output_action,
            goal_text=instruction
        )
        # 输出目录中保存 Cosmos 推理结果
        log_file = os.path.join(output_dir, "cosmos_reason1_results.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step {idx}\n")
            f.write(f"Instruction: {instruction}\n")
            f.write(f"Action sequence: {dual_sys_output.output_action}\n")
            f.write(f"Cosmos-Reason1 Result: {cosmos_result}\n")
            f.write("-" * 60 + "\n")
        print(f"[Cosmos-Reason1 推理结果已写入 {log_file}]")
        print("[Cosmos-Reason1 推理结果]:", cosmos_result)


    

    t1 = time.time()
    generate_time = t1 - t0
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="/media/agi/公共盘/InternNav-main/checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
    agent.reset()

    app.run(host='0.0.0.0', port=5801)

    
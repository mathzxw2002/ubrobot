
import argparse
import json
import os
import time
from datetime import datetime

import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from flask import Flask, jsonify, request

import cv2

from cosmos_reason_infer import CosmosReasonInfer
from robobrain_reason_infer import RoboBrainUnifiedInference

from grasp_plan import RobotArmMotionPlan

app = Flask(__name__)
output_dir = ''

@app.route("/eval_reasoning_vqa_cosmos", methods=['POST'])
def eval_cosmos_reason():

    print("eval cosmos reason ...")

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')

    instruction = data['ins']
    resut_str = cosmos_infer.infer_once(image, instruction)
    return resut_str

@app.route("/eval_reasoning_vqa", methods=['POST'])
def eval_robobrain2_5_vqa():
    print("eval robobrain 2.5 ...")

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')

    instruction = data['ins']

    resut_str = robobrain_infer.inference(instruction, image, task="general")
    print(f"Prediction:\n{resut_str}")
    return resut_str

@app.route("/eval_reasoning_traj", methods=['POST'])
def eval_robobrain2_5_traj():
    print("eval robobrain 2.5 ...")

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image_np = np.asarray(image)

    depth_pil = Image.open(depth_file.stream)
    if depth_pil.mode != 'I;16':
        raise ValueError(f" Depth Image Format Error! Expected'I;16'，Received {depth_pil.mode}")
    depth = np.asarray(depth_pil).astype(np.uint16)
    #depth = depth.convert('I')
    #depth = np.asarray(depth)
    #depth = depth.astype(np.float32) / 10000.0
    
    instruction = data['ins']

    # Visualization results will be saved to ./result, if `plot=True`. 
    # Output is formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], 
    # where each tuple contains the x and y coordinates and the depth of the point.
    resut_str = robobrain_infer.inference(instruction, image, task="trajectory", plot=False, do_sample=False)
    print(f"Prediction:\n{resut_str}")

    # check and optimize traj by pyroboplan instead of moveit
    # TODO
    # temporaly use a hard code camera intrinsics
    workspace_mask = None # TODO 
    intrinsic = None # TODO
    factor_depth = 1000.0 # TODO
    
    # save color and depth image
    image.save("./rgb.jpg", quality=95)
    save_pil = Image.fromarray(depth, mode='I;16')
    save_pil.save("./depth.png", format='PNG')

    fx = 907.7446899414062
    fy = 907.4523315429688
    cx = 644.997802734375
    cy = 369.12054443359375
    width = 1280
    height = 720
    intrinsic = np.eye(3, dtype=np.float32)

    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = cx
    intrinsic[1][2] = cy
    
    factor_depth = 1000.0
    gg = rmp.generate_6d_grasp_pose(image_np, depth, workspace_mask, intrinsic, factor_depth)
    
    # optimize the initial path given by vlm 
    
    return resut_str

@app.route("/eval_reasoning_grounding", methods=['POST'])
def eval_robobrain2_5_grounding():
    print("eval robobrain 2.5 ...")

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')

    instruction = data['ins']

    resut_str = robobrain_infer.inference(instruction, image, task="grounding", plot=False, do_sample=False)
    print(f"Prediction:\n{resut_str}")
    return resut_str

if __name__ == '__main__':

    #model_name = "/home/sany/.cache/modelscope/hub/models/nv-community/Cosmos-Reason2-8B"
    #cosmos_infer = CosmosReasonInfer(model_name)

    model_name = "/home/sany/.cache/modelscope/hub/models/BAAI/RoboBrain2.5-8B-NV"
    robobrain_infer = RoboBrainUnifiedInference(model_name)

    checkpoint_path_param = "/home/sany/ubrobot/assets/checkpoint-rs.tar"
    rmp = RobotArmMotionPlan(checkpoint_path_param)

    app.run(host='0.0.0.0', port=5802)

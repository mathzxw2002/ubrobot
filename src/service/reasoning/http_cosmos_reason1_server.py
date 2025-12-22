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

from cosmos_reason1_infer import CosmosReason1Infer

app = Flask(__name__)
output_dir = ''

@app.route("/eval_cosmos_reason1", methods=['POST'])
def eval_cosmos_reason1():

    print("eval cosmos reason ...")

    image_file = request.files['image']
    json_data = request.form['json']
    data = json.loads(json_data)

    image = Image.open(image_file.stream)
    image = image.convert('RGB')

    instruction = data['ins']
    #instruction = "Describe this video."
    resut_str = cosmos_infer.infer_once(image, instruction)
    
    return resut_str


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="/home/sany/InternNav/scripts/notebooks/checkpoints/InternVLA-N1")
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=2)

    args = parser.parse_args()

    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason1-7B"
    cosmos_infer = CosmosReason1Infer(model_name)

    app.run(host='0.0.0.0', port=5802)

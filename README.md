# ubrobot
repo for robot navigation and manipulation (WIP)

# TODOs
- [ ] add architecure for this project
- [ ] add a seperate document for learning resources, e.g. releated papers, tutorials, models, etc.
- [ ] xx


# 0. Preparation

## 0.0 Create Conda Environment

First, we should create a conda environment.

```
conda create -n ubrobot_env python=3.10
conda activate ubrobot_env

```

```

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
#import torch
#print(torch.__version__)

for linux
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#for windows
#wget https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%2Bcu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl?download=true
#pip install flash_attn-2.7.4%2Bcu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl


pip install transformers==4.51.0 diffusers==0.31.0 accelerate==1.10.1 opencv-python==4.10.0.82 pillow==10.4.0 numpy==1.26.4 gym==0.23.1
pip install imageio==2.37.0 imageio-ffmpeg==0.6.0 ftfy==6.3.1
pip install scipy matplotlib

pip install qwen_vl_utils

#for windows
#pip install triton-windows

```

We recommend to install flash-attn2 via pre-built wheel. If you have trouble with the installation, you might also skip this installation and remove the line of attn_implementation="flash_attention_2" in the model initialization.




%pip install -e ../../. # install InternNav 



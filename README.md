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
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

#pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
#import torch
#print(torch.__version__)

for linux
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#pip install flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

#for windows
#wget https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%2Bcu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl?download=true
#pip install flash_attn-2.7.4%2Bcu124torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl


#pip install transformers==4.51.0 diffusers==0.31.0 accelerate==1.10.1 opencv-python==4.10.0.82 pillow==10.4.0 numpy==1.26.4 gym==0.23.1
pip install transformers==4.51.0 diffusers==0.31.0 accelerate==1.10.1 opencv-python==4.10.0.82 pillow==10.4.0 gym==0.23.1
pip install imageio==2.37.0 imageio-ffmpeg==0.6.0 ftfy==6.3.1
pip install scipy matplotlib
pip install qwen_vl_utils

# install InternNav
cd InternNav
pip install -e .  


#for windows
#pip install triton-windows

```

We recommend to install flash-attn2 via pre-built wheel. If you have trouble with the installation, you might also skip this installation and remove the line of attn_implementation="flash_attention_2" in the model initialization.


conda install ffmpeg=7.1.1 -c conda-forge

cd lerobot
pip install -e ".[aloha, pusht]"


torchcodec	torch	Python
main / nightly	main / nightly	>=3.10, <=3.13
0.8	2.9	>=3.10, <=3.13
0.7	2.8	>=3.9, <=3.13
0.6	2.8	>=3.9, <=3.13
0.5	2.7	>=3.9, <=3.13
0.4	2.7	>=3.9, <=3.13
0.3	2.7	>=3.9, <=3.13
0.2	2.6	>=3.9, <=3.13
0.1	2.5	>=3.9, <=3.12
0.0.3	2.4	>=3.8, <=3.12

lerobot-eval --policy.path=/home/sany/.cache/modelscope/hub/models/lerobot/diffusion_pusht_migrated --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cuda

python src/lerobot/scripts/lerobot_eval.py

python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path /media/sany/ef87a074-cf12-40ba-ba8a-e4080adbba8b/modelscope/hub/models/lerobot/diffusion_pusht


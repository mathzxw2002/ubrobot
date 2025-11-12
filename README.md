# ubrobot
repo for robot navigation and manipulation (WIP)

# TODOs
- [ ] add architecure for this project
- [ ] add a seperate document for learning resources, e.g. releated papers, tutorials, models, etc.
- [ ] xx


# Installation

UBRobot works with Python 3.10+ and PyTorch 2.8+.

## Environment Setup

Create a virtual environment with Python 3.10 and activate it:

```bash
conda create -y -n lerobot python=3.10
conda activate ubrobot
```

When using conda, install ffmpeg in your environment:

```bash 
conda install ffmpeg=7.1.1 -c conda-forge
```

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install transformers==4.51.0 diffusers==0.31.0 accelerate==1.10.1 opencv-python==4.10.0.82 pillow==10.4.0 gym==0.23.1
pip install imageio==2.37.0 imageio-ffmpeg==0.6.0 ftfy==6.3.1
pip install scipy matplotlib
pip install qwen_vl_utils
```

## Install LeRobot

### From Source
First, clone the repository and navigate into the directory:
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot

pip install -e ".[aloha, pusht]"

```

## Install InternNav from Source

```

cd InternNav
pip install -e .  

```


| torchcodec       | torch           | Python          |
|------------------|-----------------|-----------------|
| main / nightly   | main / nightly  | >=3.10, <=3.13  |
| 0.8              | 2.9             | >=3.10, <=3.13  |
| 0.7              | 2.8             | >=3.9,  <=3.13  |
| 0.6              | 2.8             | >=3.9,  <=3.13  |
| 0.5              | 2.7             | >=3.9,  <=3.13  |
| 0.4              | 2.7             | >=3.9,  <=3.13  |
| 0.3              | 2.7             | >=3.9,  <=3.13  |
| 0.2              | 2.6             | >=3.9,  <=3.13  |
| 0.1              | 2.5             | >=3.9,  <=3.12  |
| 0.0.3            | 2.4             | >=3.8,  <=3.12  |



lerobot-eval --policy.path=/home/sany/.cache/modelscope/hub/models/lerobot/diffusion_pusht_migrated --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cuda

python src/lerobot/scripts/lerobot_eval.py

python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path /media/sany/ef87a074-cf12-40ba-ba8a-e4080adbba8b/modelscope/hub/models/lerobot/diffusion_pusht


# Related Projects
- [InternNav](https://github.com/InternRobotics/InternNav) : A open platform for building generalized navigation foundation models (with 6 mainstream benchmarks and 10+ baselines).
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL): The pretrained vision-language foundation model.
- [LeRobot](https://github.com/huggingface/lerobot): The data format used in this project largely follows the conventions of LeRobot.
- [Cosmos](https://github.com/nvidia-cosmos/cosmos-reason1): Cosmos-Reason1 models understand the physical common sense and generate appropriate embodied decisions in natural language through long chain-of-thought reasoning processes.

# Related Papers and Tutorials
```bibtex
@misc{internnav2025,
    title = {{InternNav: InternRobotics' open platform for building generalized navigation foundation}},
    author = {InternNav Contributors},
    howpublished = {\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
```

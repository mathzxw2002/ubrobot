
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "/home/sany/.cache/modelscope/hub/datasets/lerobot/pusht"

# 1) Load from the Hub (cached locally)
dataset = LeRobotDataset(repo_id)

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")

dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=4,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    observations = batch["observation.state"]
    actions = batch["action"]

    break

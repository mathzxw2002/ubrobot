
## 1、异步运行
### 1.1、异步运行服务器端
```bash
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=6006 --fps=30
```

### 1.2、异步运行机器人端客户端

```bash 
python -m lerobot.async_inference.robot_client --server_address=10.88.229.74:6006 --robot.type=so101_follower --robot.port=COM3 --robot.id=follower01 --robot.cameras="{ top: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 25,rotation: 'ROTATE_180'},wrist: {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 25,rotation: 'ROTATE_90'} }" --task="put" --policy_type=act --pretrained_name_or_path=D:/lerobot_new/lerobot/output/act_so101merged_records/checkpoint/080000/pretrained_model --policy_device=cpu --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --debug_visualize_queue_size=True
```
```plain
--server_address：服务器地址
--robot.port：机器人串口
--robot.id：机器人编号
--robot.cameras：机器人摄像头({摄像头名称:{摄像头参数:type:摄像头类型, index_or_path:摄像头索引或路径, width:摄像头宽度, height:摄像头高度, fps:摄像头帧率,rotation:摄像头旋转角度}）})
--task：任务名称
--policy_type：策略类型
--pretrained_name_or_path：预训练模型路径
--policy_device：策略设备
--actions_per_chunk：动作块大小
--chunk_size_threshold：块大小阈值
--aggregate_fn_name：聚合函数名称
--debug_visualize_queue_size：是否可视化队列大小
```

## 2、运行so101
### 2.1、运行机器人端
```bash
python -m lerobot.robots.so101_follower.so101_host --robot.id=my_so101 --robot.port=/dev/ttyACM0
```

robot.id：机器人编号
robot.port：机器人串口

### 2.2、运行服务器端

```bash
python examples/so101_to_so101/teleoperate_networked.py
python examples/so101_to_so101/evaluate.py
python examples/so101_to_so101/record.py
```

teleoperate_networked.py中主要的可调整参数
remote_ip：服务器地址
port：主动臂串口

evaluate.py中的主要可调整参数
remote_ip：服务器地址
HF_MODEL_ID：评估的模型路径
LOCAL_EVAL_PATH：模型评估的视频保存路径
TASK_DESCRIPTION：任务描述

record.py中的主要可调整参数
remote_ip：服务器地址
LOCAL_DATASET_PATH：录制的视频保存路径
TASK_DESCRIPTION：任务描述

## 3、数据集合并脚本

```bash
python -m lerobot.scripts.lerobot_edit_dataset --repo_id new_dataset_merged --operation.type merge --operation.repo_ids "['new_dataset1', 'new_dataset2', 'new_dataset3', 'new_dataset4', 'new_dataset5', 'new_dataset6', 'new_dataset7','new_dataset8', 'new_dataset9', 'new_dataset10', 'new_dataset11', 'new_dataset12', 'new_dataset13', 'new_dataset14', 'new_dataset15','new_dataset16', 'new_dataset17', 'new_dataset18', 'new_dataset19', 'new_dataset20']" --root "D:\lerobot_new\lerobot\data"
```

--repo_id：合并后数据集名称
--operation.type：操作类型
--operation.repo_ids：操作的数据集名称
--root：根路径，搭配operation.repo_ids组成完整的绝对路径

## 4、数据集版本转换（2.1版本的数据集格式转换为3.0版本）

```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id="D:\python\lerobot\so101-table-cleanup"  --push-to-hub=false
```

--repo-id：数据集名称
--push-to-hub：是否推送到hub

## 5、模型文件拆分，使老模型文件适应新的模型文件的格式
```bash
python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path D:/lerobot_new/lerobot/output/060000_170sample_flip/pretrained_model
```

--pretrained-path：待转换的模型文件路径

## 6、模型训练命令

```bash
lerobot-train --policy.type=act --dataset.repo_id="D:\lerobot_new\lerobot\data\so101_merged_records" --output_dir=output/train/act_from_local --job_name=act_from_local --policy.device=cpu --wandb.enable=false  --policy.push_to_hub=false
```

--policy.type：策略类型
--dataset.repo_id：数据集路径
--output_dir：模型保存路径
--job_name：模型名称
--policy.device：训练设备
--wandb.enable：是否启用wandb
--policy.push_to_hub：是否推到hub

## 7、多GPU训练

### 7.1、选取多GPU训练的配置
accelerate config

### 7.2、

```bash
accelerate launch --num_processes=8 --num_machines=1 --mixed_precision=no -- /usr/local/bin/lerobot-train --policy.type=act --dataset.repo_id="/lerobot_training/so101_merged_records" --output_dir=/lerobot_training/outputs/so101-table-cleanup_8gpu --job_name=act_8GPU --policy.device=cuda --wandb.enable=false --policy.push_to_hub=false --batch_size=2
```

--num_processes：GPU数量
--num_machines：机器数量
--mixed_precision：混合精度
/usr/local/bin/lerobot-train：训练命令
--policy.type：策略类型
--dataset.repo_id：数据集路径
--output_dir：模型保存路径
--job_name：模型名称
--policy.device：训练设备
--wandb.enable：是否启用wandb
--policy.push_to_hub：是否推到hub
--batch_size：每个GPU的批次大小

## 8、查找相机串口
lerobot-find-cameras opencv


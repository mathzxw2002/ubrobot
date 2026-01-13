      
# cosmos_reason1_infer.py
from pathlib import Path
import torch
import transformers
import qwen_vl_utils

SEPARATOR = "-" * 20


class CosmosReason1Agent:
    def __init__(self, model_path: str):
        print("[Cosmos-Reason1] Loading model...")
        self.model_path = model_path
        self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor: transformers.Qwen2_5_VLProcessor = transformers.AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        print("[Cosmos-Reason1] Model loaded successfully ✅")

    @torch.no_grad()
    def infer_once(self, image_path: str, action_seq, goal_text: str):
        """
        image_path: 图像路径
        action_seq: 当前动作序列（list 或 str）
        goal_text: 目标描述（str）
        """

        # 格式化动作序列
        if isinstance(action_seq, (list, tuple)):
            action_seq_str = str(action_seq)
        else:
            action_seq_str = action_seq

        # ==== 构造完整的 prompt ====
        conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text":
                "You are an expert assistant for robotic vision-based navigation reasoning and manipulation planning."
                "Your task is to analyze the robot's current camera image, a given sequence of discrete navigation actions (where 0, 1, 2, 3 correspond to 'stop,' 'forward,' 'turn left,' and 'turn right,' respectively), and the final navigation or grasping goal."
                "0 indicates that the destination has been reached and movement should stop, 1 indicates moving forward by 0.25 meters (replace with meters if you know the step length), "
                "and 2/3 indicates turning 15 degrees to the left/right on the spot (unless other robot parameters are provided later)."
                "The action sequence represents intermediate moves toward the goal, not the full route. "
                "The red dot on the image indicates the target position for movement at this stage."
                "You must reason about whether the sequence is safe and effective based on visual cues, potential obstacles, and orientation, and decide if adjustments are needed."

                # ==== 新增逻辑开始 ====
                "After evaluating the navigation sequence, also consider a subsequent grasping (manipulation) phase."
                "Assume that after executing the movement sequence, the robot will attempt to grasp a specified target object visible in the scene (the grasp target is described in the goal)."
                "You must assess whether, from the expected final position after navigation, the robotic arm can safely and effectively reach and grasp the object."
                "Consider reachability, visibility, occlusion, object distance, and the presence of obstacles that may block the arm or gripper."
                "If grasping seems infeasible, propose minimal adjustments such as fine position changes or orientation corrections."
                # ==== 新增逻辑结束 ====

                "Follow these rules:\n"
                "1. First give a short 1-2 sentence summary of your conclusion (e.g., sequence is safe / needs adjustment / grasping may fail).\n"
                "2. Describe the current visual scene from the robot’s perspective in a clear and detailed way. Focus on spatial layout, surrounding objects, and information useful for navigation and grasping."
                "Identify the main subjects, their appearance, size, and relative positions (e.g., in front of, on the left, far away). Describe the environment type, lighting, and ground surface. Mention open or blocked pathways, obstacles, and notable landmarks. Use concise, factual, and structured sentences that support robot navigation and situational awareness.\n"
                "3. Then explain your reasoning briefly (key observations, visible obstacles, spatial layout, and grasp feasibility considerations).\n"
                "4. Finally, output a JSON block strictly following this schema:\n"
                "{\n"
                "  'verdict': 'ok' | 'needs_adjustment' | 'insufficient',\n"
                "  'vision_caption': 'detailed description of the current field of vision',\n"
                "  'summary': 'short summary',\n"
                "  'reasoning': ['list', 'of', 'points'],\n"
                "  'feasibility': {\n"
                "     'can_reach_with_sequence': true/false,\n"
                "     'estimated_remaining_steps': int or null,\n"
                "     'estimated_uncertainty_reasons': ['optional list']\n"
                "  },\n"
                "  'grasping_feasibility': {\n"
                "     'can_grasp_target': true/false,\n"
                "     'issues': ['occluded', 'too_far', 'unreachable_angle', 'object_not_visible', ...],\n"
                "     'suggested_adjustments': ['move closer', 'rotate base', 'adjust gripper orientation', ...]\n"
                "  },\n"
                "  'recommended_action_sequence': ['forward','left','forward'],\n"
                "  'adjustments': ['suggested navigation or grasping changes if needed'],\n"
                "  'confidence': 0.0-1.0,\n"
                "}\n"
                "After the JSON, output a very short natural-language summary (<=30 Chinese characters) suitable for TTS playback.\n"
                "Answer in chinese.\n\n"
                "Be concise, structured, and avoid speculation beyond the visible scene and answer in Chinese."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text":
                f"当前动作序列: {action_seq_str}\n"
                f"最终导航目标: {goal_text}\n"
                f"最终抓取目标: 抓住前面的纸杯并把它放在右边\n"
                "每步步长: 0.25 米\n"
                "转向角度: 15 度\n"
                "传感器信息: RGB 摄像头，无深度\n"
                "环境约束: 禁止发生碰撞，机械臂在移动完成后进行抓取\n\n"
                "请根据图像和以上信息，判断移动与抓取阶段的可行性，并按 JSON 输出结果与简短总结。用中文回答!"
            }
        ],
    },
]


        # ==== 前处理 ====
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # ==== 推理 ====
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        result = output_text[0]
        # print(SEPARATOR)
        # print(result)
        # print(SEPARATOR)
        return result

    
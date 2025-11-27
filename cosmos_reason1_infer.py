# cosmos_reason1_infer.py
from pathlib import Path
import torch
import transformers
import qwen_vl_utils

SEPARATOR = "-" * 20


class CosmosReason1Infer:
    def __init__(self, model_path: str):
        print("[Cosmos-Reason1] Loading model...")
        # Load model
        self.model_path = model_path
        #model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason1-7B"
        self.model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="float16", device_map="auto"
        )
        self.processor: transformers.Qwen2_5_VLProcessor = transformers.AutoProcessor.from_pretrained(
            model_path, local_files_only=True
        )
        print("[Cosmos-Reason1] Model loaded successfully ✅")

    @torch.no_grad()
    def infer_once(self, image_path: str, infer_instruct_str: str):
        """
        image_path: path for the image
        infer_instruct_str: task instruction
        """

        # Create inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"./sample.mp4",
                        "fps": 4,
                        # 6422528 = 8192 * 28**2 = vision_tokens * (2*spatial_patch_size)^2
                        "total_pixels": 6422528,
                    },
                    {"type": "text", "text": infer_instruct_str}, #"Describe this video."
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Run inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        print(SEPARATOR)
        print(output_text[0])
        print(SEPARATOR)

        result = output_text[0]
        return result
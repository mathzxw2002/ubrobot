# cosmos_reason_infer.py
from pathlib import Path
import torch
import transformers

import cv2

from PIL import Image as PIT_Image

SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


class CosmosReasonInfer:
    def __init__(self, model_path: str):
        # Ensure reproducibility
        transformers.set_seed(0)
        print("[Cosmos-Reason2] Loading model...")
        
        # Load model
        self.model_path = model_path
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=torch.float16, device_map="auto"
        )
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(self.model_path)

        # Optional: Limit vision tokens
        min_vision_tokens = 256
        max_vision_tokens = 8192
        self.processor.image_processor.size = {
            "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
        }
        self.processor.video_processor.size = {
            "shortest_edge": min_vision_tokens * PIXELS_PER_TOKEN,
            "longest_edge": max_vision_tokens * PIXELS_PER_TOKEN,
        }
        print("[Cosmos-Reason] Model loaded successfully ✅")

    @torch.no_grad()
    def infer_once(self, img: PIT_Image, infer_instruct_str: str):
        """
        image_path: path for the image
        infer_instruct_str: task instruction
        """

        print("new infer onece ===================", infer_instruct_str)

        # Create inputs
        # IMPORTANT: Media is listed before text to match training inputs
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                        "total_pixels": 6422528,
                    },
                    {"type": "text", "text": infer_instruct_str},
                ],
            },
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            #fps=4,
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

if __name__ == '__main__':

    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason2-8B"
    cosmos_infer = CosmosReasonInfer(model_name)

    
    image = cv2.imread("./segment_result.jpg", flags=cv2.IMREAD_COLOR)
    instruction = "please recognize the objects in this picture, and give the coordinate of each object."
    resut_str = cosmos_infer.infer_once(image, instruction)
    print(resut_str)
    

import torch

import cv2

from PIL import Image as PIT_Image

from cosmos_reason_infer import CosmosReasonInfer
from robobrain_reason_infer import RoboBrainUnifiedInference


class VLMReasonInfer:
    def __init__(self, model_path: str):
        # Load model
        self.model_path = model_path
        #self.vlm_reason_infer = RoboBrainUnifiedInference(self.model_path)
        self.vlm_reason_infer = CosmosReasonInfer(self.model_path)
        print("[VLM Reason] Model loaded successfully ✅")

    @torch.no_grad()
    def infer_once(self, img: PIT_Image, infer_instruct_str: str):
        """
        image_path: path for the image
        infer_instruct_str: task instruction
        """
        print("new infer onece ===================", infer_instruct_str)
        result = self.vlm_reason_infer.infer_once(img, infer_instruct_str)
        return result

if __name__ == '__main__':

    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason2-8B"
    cosmos_infer = CosmosReasonInfer(model_name)
    
    image = cv2.imread("./segment_result.jpg", flags=cv2.IMREAD_COLOR)
    instruction = "please recognize the objects in this picture, and give the coordinate of each object."
    resut_str = cosmos_infer.infer_once(image, instruction)
    print(resut_str)
    

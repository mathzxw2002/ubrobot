import gradio as gr
import random
import time

from pathlib import Path

import qwen_vl_utils
import transformers

ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20


#CosmosReason1Infer cosmos_infer

with gr.Blocks() as demo:

    # Load model
    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason1-7B"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="float16", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

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
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = output_text[0] #random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(server_name="0.0.0.0", server_port=7861)
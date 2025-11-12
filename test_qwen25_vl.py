import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

from qwen_plot_util import plot_bounding_boxes_abs

# 1. Load the model and processor
model_path = "/home/sany/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct" # Use an appropriate model size
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    #attn_implementation="flash_attention_2" # Recommended for speed and memory
)
processor = AutoProcessor.from_pretrained(model_path)

# 2. Load the image
# Replace with your local image path
image_path = "./assets/dining_table.png"
image = Image.open(image_path).convert("RGB")

# 3. Define the prompt for grounding
# The prompt should ask for the bounding box in a structured way.
# Qwen-VL expects absolute coordinates.
# The model will output bounding box tokens, typically in the format:
# "<box>x_min, y_min, x_max, y_max</box>"
prompt_text = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'

# Structure the messages according to the chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }
]

# 4. Prepare inputs for the model
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# 5. Run inference
generated_ids = model.generate(**inputs, max_new_tokens=512)

# 6. Decode the output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

print(output_text)

# The output_text will contain the bounding box coordinates, e.g.,
# "The cat is located at <box>120, 250, 400, 600</box>."
# You will need to parse this string to extract the coordinates for further use (e.g., drawing the box).



'''prompt = 'locate every instance that belongs to the following categories: "plate/dish, scallop, wine bottle, tv, bowl, spoon, air conditioner, coconut drink, cup, chopsticks, person". Report bbox coordinates in JSON format.'
img_url = "./assets/spatial_understanding/dining_table.png"
model_response = inference_with_openai_api(img_url, prompt)
print(model_response)

response = requests.get(img_url)
response.raise_for_status()
image = Image.open(BytesIO(response.content))

image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, model_response)'''

#image.thumbnail([640,640], Image.Resampling.LANCZOS)

plot_bounding_boxes_abs(image, output_text)

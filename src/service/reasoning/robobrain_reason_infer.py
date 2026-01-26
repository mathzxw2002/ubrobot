import os, re, cv2
from typing import Union
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

class RoboBrainUnifiedInference:
    """
    A unified class for performing inference using RoboBrain 2.5 models.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain2.5-8B-NV", device_map="auto"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.)
        """
        print("Loading Checkpoint ...")
        self.model_id = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype="auto", 
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def inference(self, text: str, image: Union[list, str], task="general", 
                 plot=False, do_sample=True, temperature=0.7):
        """
        Perform inference with text and images input.
        
        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) as a list of file paths or a single file path.
            task (str): The task type, e.g., "general", "pointing", "trajectory", "grounding".
            plot (bool): Whether to plot results on image.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
        """

        if isinstance(image, str):
            image = [image]

        assert task in ["general", "pointing", "trajectory", "grounding"], \
            f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'trajectory', 'grounding'."
        assert task == "general" or (task in ["pointing", "trajectory", "grounding"] and len(image) == 1), \
            "Pointing, grounding, and trajectory tasks require exactly one image."

        if task == "pointing":
            print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], where the tuple contains the x and y coordinates of a point satisfying the conditions above."
        elif task == "trajectory":
            print("Trajectory task detected. Adding trajectory prompt.")
            text = f"Please predict 3D end-effector-centric waypoints to complete the task successfully. The task is \"{text}\". Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the depth of the point."
        elif task == "grounding":
            print("Grounding task detected. Adding grounding prompt.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", 
                         "image": path if path.startswith("http") else f"file://{path}"
                        } for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        print("Running inference ...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer_text = output_text[0] if output_text else ""

        # Plotting functionality
        if plot and task in ["pointing", "trajectory", "grounding"]:
            print("Plotting enabled. Drawing results on the image ...")
            
            plot_points, plot_boxes, plot_trajectories = None, None, None
            result_text = answer_text  # Use the processed answer text for plotting
            
            if task == "trajectory":
                trajectory_pattern = r'(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)'
                trajectory_points = re.findall(trajectory_pattern, result_text)
                plot_trajectories = [[(int(x), int(y), float(d)) for x, y, d in trajectory_points]]
                print(f"Extracted trajectory points: {plot_trajectories}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_trajectory_annotated.")
            elif task == "pointing":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                points = re.findall(point_pattern, result_text)
                plot_points = [(int(x), int(y)) for x, y in points]
                print(f"Extracted points: {plot_points}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_pointing_annotated.")
            elif task == "grounding":
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                boxes = re.findall(box_pattern, result_text)
                plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_grounding_annotated.")

            os.makedirs("result", exist_ok=True)
            image_path_to_save = os.path.join("result", image_name_to_save)

            self.draw_on_image(
                image[0], 
                points=plot_points, 
                boxes=plot_boxes, 
                trajectories=plot_trajectories,
                output_path=image_path_to_save
            )

        # Return unified format
        result = {"answer": answer_text}
        return result

    def draw_on_image(self, image_path, points=None, boxes=None, trajectories=None, output_path=None):
        """
        Draw points, bounding boxes, and trajectories on an image

        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x, y), ...] where x,y are relative (0~1000)
            boxes: List of boxes in format [[x1, y1, x2, y2], ...] where coords are relative (0~1000)
            trajectories: List of trajectories in format [[(x, y), (x, y), ...], ...]
                        or [[(x, y, d), ...], ...] where x,y are relative (0~1000)
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                """Convert relative (0~1000) to absolute pixel coords, clamped to image bounds."""
                x = int(round((x_rel / 1000.0) * w))
                y = int(round((y_rel / 1000.0) * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            # Draw points
            if points:
                for point in points:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1r, y1r, x2r, y2r = box
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if not trajectory or len(trajectory) < 2:
                        continue

                    # Convert all trajectory points to absolute pixels
                    abs_pts = []
                    for p in trajectory:
                        # support (x,y) or (x,y,d)
                        x_rel, y_rel = p[0], p[1]
                        abs_pts.append(rel_to_abs(x_rel, y_rel))

                    # Connect trajectory points with lines
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 0, 255), 2)  # Blue line

                    # Draw a larger point at the trajectory end
                    start_x, start_y = abs_pts[0]
                    cv2.circle(image, (start_x, start_y), 7, (0, 255, 0), -1)  # Red start point

                    # Draw a larger point at the trajectory end
                    end_x, end_y = abs_pts[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)  # Blue end point

            # Determine output path
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            # Save the result
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


# Usage examples
if __name__ == "__main__":
    print("=== Testing RoboBrain2.5-8B-NV Model ===")
    model_8b = UnifiedInference("BAAI/RoboBrain2.5-8B-NV")
    # Case 1
    prompt = "What is shown in this image?"
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"
    pred_8b = model_8b.inference(prompt, image, task="general")
    print(f"Prediction:\n{pred_8b}")
    # Case 2
    prompt = "the person wearing a red hat"
    image = "./assets/demo/grounding.jpg"
    pred_8b = model_8b.inference(prompt, image, task="grounding", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
    # Case 3
    prompt = "the affordance area for holding the cup"
    image = "./assets/demo/affordance.jpg"
    pred_8b = model_8b.inference(prompt, image, task="pointing", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
    # Case 4
    prompt = "reach for the banana on the plate"
    image = "./assets/demo/trajectory.jpg"
    pred_8b = model_8b.inference(prompt, image, task="trajectory", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
    # Case 5
    prompt = "Identify spot within the vacant space that's between the two mugs"
    image = "./assets/demo/pointing.jpg"
    pred_8b = model_8b.inference(prompt, image, task="pointing", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
    # Case 6
    prompt = "Identify spot within toilet in the house"
    image = "./assets/demo/navigation1.jpg"
    pred_8b = model_8b.inference(prompt, image, task="pointing", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
    # Case 7
    prompt = "Identify spot within sofa in the house"
    image = "./assets/demo/navigation2.jpg"
    pred_8b = model_8b.inference(prompt, image, task="pointing", plot=True, do_sample=False)
    print(f"Prediction:\n{pred_8b}")
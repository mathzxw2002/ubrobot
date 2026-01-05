from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model("./bus.jpg")  # Predict on an image
#results[0].show()  # Display results

# Export the model to ONNX format for deployment
#path = model.export(format="onnx")  # Returns the path to the exported model

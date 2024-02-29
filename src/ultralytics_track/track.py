from ultralytics import YOLO
import os

# Load YOLOV8 Model
model_filename = 'anka_v1.pt'
model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
model = YOLO(model_path)

source_path = os.path.join(os.path.dirname(__file__), "balloon_video.mp4") # Path to video, device id (int, usually 0 for built in webcams)

# Show and save results to project folder
for r in model.track(source=source_path, show=True, stream=True, persist=True, save=True, project=os.path.join(os.path.dirname(__file__), "val")):
    pass
# DEPRECATED: results = model.track(source=source_path, show=True, persist=True, save=True, project=os.path.join(os.path.dirname(__file__), "val"))

# TODO: Put points on the center of the bounding boxes
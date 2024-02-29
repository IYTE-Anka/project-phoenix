from ultralytics import YOLO
import os

# Load YOLOV8 Model
model_filename = 'anka_v1.pt'
model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
model = YOLO(model_path)

source_path = os.path.join(os.path.dirname(__file__), "balloon_video.mp4") # Path to video, device id (int, usually 0 for built in webcams)
results = model.track(source=source_path, show=True, persist=True, save=True, project=os.path.dirname(__file__)) # Save results to project folder

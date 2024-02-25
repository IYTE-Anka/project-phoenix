from ultralytics import YOLO
import cv2
import os

# Load YOLOV8 Model
model_filename = 'anka_v1.pt'
model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
model = YOLO(model_path)

results = model.track(source=0, show=True)
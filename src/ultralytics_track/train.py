from ultralytics import YOLO
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

def download_dataset():
  rf = Roboflow(api_key=ROBOFLOW_API_KEY)
  project = rf.workspace("prasku-mxsdv").project("blimp_yolo_v8_custom")
  version = project.version(1)
  dataset = version.download("yolov8")

def train():
  model = YOLO('yolov8s.pt')
  results = model.train(data='datasets\Blimp_Yolo_v8_Custom-1\data.yaml', epochs=20, imgsz=640, device=0)

if __name__=="__main__":
  train()
  print("Training completed.")
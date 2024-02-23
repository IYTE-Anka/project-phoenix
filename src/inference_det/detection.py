import inference
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes
import os
from dotenv import load_dotenv

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="blimp_yolo_v8_custom/1", # Roboflow model to use
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=render_boxes, # Function to run after each prediction
    api_key=ROBOFLOW_API_KEY, 
)
pipeline.start()
pipeline.join()



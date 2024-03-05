from collections import defaultdict
import cv2
import numpy as np
import os
from ultralytics import YOLO
import argparse

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--source', 
      help='Path of the video or camera id.', 
      required=False, 
      default=os.path.join(os.path.dirname(__file__), "testing", "red_v1.mp4")
    )
  parser.add_argument(
      '--model', 
      help='Path of the object detection model.', 
      required=False, 
      default=os.path.join(os.path.dirname(__file__), "models", "anka_v1.0.pt")
    )
  parser.add_argument(
      '--conf', 
      help='Path of the object detection model.', 
      required=False, 
      default=0.30
    )
  args = parser.parse_args()

  source_path = os.path.join(os.path.dirname(__file__), "testing", args.source)
  model_path = os.path.join(os.path.dirname(__file__), "models", args.model)
  conf_threshold = float(args.conf)
  
  run(source_path, model_path, conf_threshold)

def run(source_path, model_path, conf_threshold):
  # Load the YOLOv8 model
  model = YOLO(model_path)

  # Open the video file
  cap = cv2.VideoCapture(source_path)

  # Store the track history
  track_history = defaultdict(lambda: [])

  # Loop through the video frames
  while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
      # Run YOLOv8 tracking on the frame, persisting tracks between frames
      results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold)

      if results[0].boxes is None or results[0].boxes.id is None: # Prevent the program from crashing when no object is detected
        cv2.imshow("ANKA Balon Tespit", frame)
        continue

      # Get the boxes and track IDs
      boxes = results[0].boxes.xywh.cpu()
      track_ids = results[0].boxes.id.int().cpu().tolist()

      # Visualize the results on the frame
      annotated_frame = results[0].plot()

      # Plot the tracks
      for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
          track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

      # Display the annotated frame
      cv2.imshow("ANKA Balon Tespit", annotated_frame)

      # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else:
      # Break the loop if the end of the video is reached
      break

  # Release the video capture object and close the display window
  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
from collections import defaultdict
from time import sleep
from ultralytics import YOLO
import cv2
import numpy as np
import os
import argparse
import socket
import struct
import io
import sys

try:
  # Replace with your Raspberry Pi's IP address
  HOST = str(input("Please enter the current Raspberry Pi 4 IP address (enter 0 if not needed): "))
  PORT = 8000

  if HOST == "0":
    print("Raspberry Pi 4 connection not needed, proceeding with local detection...")
    sleep(1)
  else: 
    # Create a socket to receive the video
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    connection = client_socket.makefile('rb')
    print(f"Connection established to {HOST}")
    sleep(1)

except Exception as e:
  print("Following error occured: ", e)
  sleep(1)
  sys.exit("Exiting...")


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
      default=os.path.join(os.path.dirname(__file__), "models", "anka_v1.2.pt")
    )
  parser.add_argument(
      '--conf', 
      help='Confidence threshold for detection.', 
      required=False, 
      default=0.30
    )
  parser.add_argument(
      '--color', 
      help='Enables color tracking.',
      dest="color",
      required=False, 
      action="store_true"
    )
  parser.add_argument(
      '--no-color', 
      help='Disable color tracking.', 
      dest="color",
      required=False, 
      action="store_false"
    )
  parser.set_defaults(color=True)

  args = parser.parse_args()
  source_path = ""
  try:
    if int(args.source) == 0:
      source_path = 0
  except ValueError:
    source_path = os.path.join(os.path.dirname(__file__), "testing", args.source) if str(args.source) != "pi" else "pi"

  model_path = os.path.join(os.path.dirname(__file__), "models", args.model)
  conf_threshold = float(args.conf)
  
  if source_path == "pi":
    run_pi("pi", model_path, conf_threshold, args.color)
  else:
    run(source_path, model_path, conf_threshold, args.color)

def run(source_path, model_path, conf_threshold, color):
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
      # SECTION: OBJECT DETECTION
      # Run YOLOv8 tracking on the frame, persisting tracks between frames
      results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, device=0)

      # Visualize the results on the frame
      annotated_frame = results[0].plot()
      boxes = None
      # SECTION: TRAJECTORY PLOTTING
      if results[0].boxes is not None and results[0].boxes.id is not None: # Fixes Issue#13 - Video stops in the output when there is no detection 
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

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
          cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 0), -1) # Put circle on the center of the balloons bboxes

      # SECTION: COLOR TRACKING
      if color: # Check if color tracking is enabled
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Set range for red color and  
        # define mask 
        red_lower = np.array([136, 87, 111], np.uint8) 
        red_upper = np.array([180, 255, 255], np.uint8) 
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
        kernel = np.ones((5, 5), "uint8") 

        # For red color 
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red = cv2.bitwise_and(frame, frame, mask = red_mask) 

        # Creating contour to track red color 
        red_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(red_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (0, 0, 255), 2)
                  cv2.putText(annotated_frame, "Kirmizi", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)) 

        # Set range for green color and  
        # define mask 
        green_lower = np.array([40, 40, 40], np.uint8) 
        green_upper = np.array([70, 255, 255], np.uint8) 
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

        # For green color 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(frame, frame, mask = green_mask) 

        # Creating contour to track green color 
        green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(green_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (0, 255, 0), 2)
                  cv2.putText(annotated_frame, "Yesil", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))

        # Set range for blue color and  
        # define mask 
        blue_lower = np.array([100, 150, 0], np.uint8) 
        blue_upper = np.array([140, 255, 255], np.uint8) 
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

        # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask) 

        # Creating contour to track blue color 
        blue_contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(blue_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (255, 0, 0), 2)
                  cv2.putText(annotated_frame, "Mavi", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))

      # Display FPS
      fps = str(int(cap.get(cv2.CAP_PROP_FPS)))
      cv2.putText(annotated_frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

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

def run_pi(source_path, model_path, conf_threshold, color):
  # Load the YOLOv8 model
  model = YOLO(model_path)

  # Store the track history
  track_history = defaultdict(lambda: [])

  # Loop through the video frames
  while True:

    success = True
    # Read the length of the image as a 32-bit unsigned int. If the length is zero, break
    image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
    if not image_len:
        break
    # Construct a stream to hold the image data and read the image data from the connection
    image_stream = io.BytesIO()
    image_stream.write(connection.read(image_len))
    image_stream.seek(0)
    # Decode the image from the stream
    image = np.asarray(bytearray(image_stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Read a frame from the video
    frame = image

    if success:
      # SECTION: OBJECT DETECTION
      # Run YOLOv8 tracking on the frame, persisting tracks between frames
      results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, device=0)

      # Visualize the results on the frame
      annotated_frame = results[0].plot()
      boxes = None
      # SECTION: TRAJECTORY PLOTTING
      if results[0].boxes is not None and results[0].boxes.id is not None: # Fixes Issue#13 - Video stops in the output when there is no detection 
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

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
          cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 0), -1) # Put circle on the center of the balloons bboxes

      # SECTION: COLOR TRACKING
      if color: # Check if color tracking is enabled
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Set range for red color and  
        # define mask 
        red_lower = np.array([136, 87, 111], np.uint8) 
        red_upper = np.array([180, 255, 255], np.uint8) 
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
        kernel = np.ones((5, 5), "uint8") 

        # For red color 
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red = cv2.bitwise_and(frame, frame, mask = red_mask) 

        # Creating contour to track red color 
        red_contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(red_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (0, 0, 255), 2)
                  cv2.putText(annotated_frame, "Kirmizi", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255)) 

        # Set range for green color and  
        # define mask 
        green_lower = np.array([40, 40, 40], np.uint8) 
        green_upper = np.array([70, 255, 255], np.uint8) 
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

        # For green color 
        green_mask = cv2.dilate(green_mask, kernel) 
        res_green = cv2.bitwise_and(frame, frame, mask = green_mask) 

        # Creating contour to track green color 
        green_contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(green_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (0, 255, 0), 2)
                  cv2.putText(annotated_frame, "Yesil", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))

        # Set range for blue color and  
        # define mask 
        blue_lower = np.array([100, 150, 0], np.uint8) 
        blue_upper = np.array([140, 255, 255], np.uint8) 
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

        # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(frame, frame, mask = blue_mask) 

        # Creating contour to track blue color 
        blue_contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(blue_contours): 
          area = cv2.contourArea(contour) 
          if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            if boxes is not None:
              for box, track_id in zip(boxes, track_ids): # Fixes Issue#14 - Color detection running in unwanted sections
                Bx, By, Bw, Bh = box
                if abs(Bx-Bw) < x < abs(Bx+Bw) and abs(By-Bh) < y < abs(By+Bh):
                  annotated_frame = cv2.rectangle(annotated_frame, (x, y),  (x + w, y + h), (255, 0, 0), 2)
                  cv2.putText(annotated_frame, "Mavi", (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))

      # Display FPS
      fps = "30" #FIXME: Temporary value
      cv2.putText(annotated_frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

      # Display the annotated frame
      cv2.imshow("ANKA Balon Tespit", annotated_frame)

      # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    else:
      # Break the loop if the end of the video is reached
      break

  # Release the video capture object and close the display window
  cv2.destroyAllWindows()
  connection.close()
  client_socket.close()

if __name__ == "__main__":
  main()
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import defaultdict
import tkinter as tk
import cv2
import socket
import numpy as np
import struct
import io
import threading
import os
import sys
import json

# Replace with your Raspberry Pi's IP address
HOST = '10.8.54.35'
PORT = 8000

current_mode = "Mod 1"
def update_mode(new_mode):
    global current_mode
    current_mode = new_mode
    print(f"Mode changed to: {current_mode}")

# Create a socket to receive the video
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('rb')

def send_data(data):
    try:
        data_bytes = data.encode('utf-8')
        size = len(data_bytes)
        client_socket.sendall(struct.pack('<L', size) + data_bytes)
        print(f"Sent data: {data}")
    except Exception as e:
        print(f"Error sending data: {e}")


def video_stream():
  global current_mode
  
  model_path = os.path.join(os.path.dirname(__file__), "ultralytics_track", "models", "anka_v1.2.pt")
  model = YOLO(model_path)

  track_history = defaultdict(lambda: [])
  try:
    while True:
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
      # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # DEPRECATED: BGR to RGB conversion is not needed anymore
      
      if current_mode in ["Mod 2", "Mod 3"]:
        # SECTION: OBJECT DETECTION
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(image, persist=True, tracker="bytetrack.yaml", conf=0.3, device=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        boxes = None
        objects = []
        
        # SECTION: TRAJECTORY PLOTTING
        if results[0].boxes is not None and results[0].boxes.id is not None: # Fixes Issue#13 - Video stops in the output when there is no detection 
          # Get the boxes and track IDs
          boxes = results[0].boxes.xywh.cpu()
          track_ids = results[0].boxes.id.int().cpu().tolist()
            
          # Plot the tracks
          for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            object_data = {
              'track_id': track_id,
              'x': int(x),
              'y': int(y),
              'w': int(w),
              'h': int(h)
            }
            objects.append(object_data)

            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
              track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 0, 0), -1) # Put circle on the center of the balloons bboxes

        if current_mode == "Mod 3":
          # SECTION: COLOR TRACKING
          # Define the color ranges for red, green, and blue in HSV color space
          color_ranges = {
            "Kirmizi": [(136, 87, 111), (180, 255, 255)],
            "Yesil": [(25, 52, 72), (102, 255, 25)],
            "Mavi": [(94, 80, 2), (120, 255, 25)]
          }

          # Convert the frame to HSV color space
          hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          if boxes is not None:
            for box, track_id in zip(boxes, track_ids):
              x, y, w, h = box
              # Extract the ROI from the HSV frame
              roi = hsv_frame[int(y):int(y+h), int(x):int(x+w)]
              for color_, (lower, upper) in color_ranges.items():
                # Create a mask for the current color
                mask = cv2.inRange(roi, lower, upper)
                # If the color is found in the ROI, annotate it on the frame
                if cv2.countNonZero(mask) > 0:
                  cv2.putText(annotated_frame, color_, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                  for obj in objects:
                    if obj['track_id'] == track_id:
                      obj['color'] = color_
                      break

        json_data = json.dumps(objects)
        send_data(json_data) if objects else send_data("WARNING: No objects detected.")

        # Display FPS
        fps = "30" #FIXME: Temporary value
        cv2.putText(annotated_frame, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated_frame, current_mode, (annotated_frame.shape[1] - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        # Convert the Image object into a TkPhoto object
        im = Image.fromarray(annotated_frame)
        img = ImageTk.PhotoImage(image=im)
        # Update the image_label with a new image
        image_label.config(image=img)
        image_label.image = img
        
      elif current_mode == "Mod 1":
        fps = "30" #FIXME: Temporary value
        cv2.putText(image, fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, current_mode, (image.shape[1] - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        # Display the frame
        im = Image.fromarray(image)
        img = ImageTk.PhotoImage(image=im)
        # Update the image_label with a new image
        image_label.config(image=img)
        image_label.image = img
  finally:
    connection.close()
    client_socket.close()

def on_key_press(event):
    if event.char == 'q':  # Check if 'q' was pressed
        try:
            connection.close()
            client_socket.close()
            sys.exit(0)
        except Exception as e:
            print(e)

def move_up():
    print("Move Up")

def move_down():
    print("Move Down")

def move_left():
    print("Move Left")

def move_right():
    print("Move Right")

root = tk.Tk()
root.bind('<KeyPress>', on_key_press)  # Bind the key press event to the on_key_press function
root.title("İYTE ANKA - Balon Tespit ve İmha")

# Mode selection frame
mode_frame = tk.Frame(root)
mode_frame.pack(side=tk.TOP, fill=tk.X)

# Mode buttons
mode1_button = tk.Button(mode_frame, text="Mod 1", command=lambda: update_mode("Mod 1"))
mode1_button.pack(side=tk.LEFT, padx=5)

mode2_button = tk.Button(mode_frame, text="Mod 2", command=lambda: update_mode("Mod 2"))
mode2_button.pack(side=tk.LEFT, padx=5)

mode3_button = tk.Button(mode_frame, text="Mod 3", command=lambda: update_mode("Mod 3"))
mode3_button.pack(side=tk.LEFT, padx=5)

# Motor control frame
control_frame = tk.Frame(root)
control_frame.pack(side=tk.TOP, fill=tk.X)

# Control buttons
up_button = tk.Button(control_frame, text="Yukarı", command=move_up)
up_button.pack(side=tk.TOP, padx=5, pady=5)

down_button = tk.Button(control_frame, text="Aşağı", command=move_down)
down_button.pack(side=tk.BOTTOM, padx=5, pady=5)

left_button = tk.Button(control_frame, text="Sol", command=move_left)
left_button.pack(side=tk.LEFT, padx=200, pady=5)

right_button = tk.Button(control_frame, text="Sağ", command=move_right)
right_button.pack(side=tk.RIGHT, padx=200, pady=5)

image_label = tk.Label(root)  
image_label.pack()  

thread = threading.Thread(target=video_stream)
thread.start()

root.mainloop()
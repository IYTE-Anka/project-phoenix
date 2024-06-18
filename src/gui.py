from PIL import Image, ImageTk
from time import sleep
from ultralytics import YOLO
from collections import defaultdict
import cv2
import tkinter as tk
import socket
import struct
import io
import sys
import os

class VideoCapture:
  def __init__(self, video_source=0):
    self.vid = cv2.VideoCapture(video_source)
    if not self.vid.isOpened():
      raise ValueError("Unable to open video source", video_source)

  def get_frame(self):
    if self.vid.isOpened():
      ret, frame = self.vid.read()
      if ret:
        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      else:
        return (ret, None)
    else:
      return (ret, None)

  def __del__(self):
    if self.vid.isOpened():
      self.vid.release()

class App:
  def __init__(self, window, window_title, video_source=0, get_frame_func=None):
    self.window = window
    self.window.title(window_title)
    self.video_source = video_source
    self.get_frame_func = get_frame_func
    self.vid = VideoCapture(self.video_source)
    self.canvas = tk.Canvas(window, width = self.vid.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height = self.vid.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.canvas.pack()
    self.delay = 15
    self.update()
    self.window.mainloop()

  def update(self):
    if self.get_frame_func:
      frame = self.get_frame_func()
    else:
      ret, frame = self.vid.get_frame()
      if not ret:
        return
    self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
    self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
    self.window.after(self.delay, self.update)

def run():
  # Load the YOLOv8 model
  model_path = os.path.join(os.path.dirname(__file__), "ultralytics_track", "models", "anka_v1.2.pt")
  model = YOLO(model_path)

  # Store the track history
  track_history = defaultdict(lambda: [])

def main():
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
  App(tk.Tk(), "ANKA Balon Tespit Sistemi", 0, run)

if __name__ == "__main__":
  main()
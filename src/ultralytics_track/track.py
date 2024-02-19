from ultralytics import YOLO
import cv2
import os


# load yolov8 model
model_filename = 'yolov8n.pt'
model_path = os.path.join(os.path.dirname(__file__), "models", model_filename)
model = YOLO(model_path)

# load video
video_path = 0
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()
    
    if ret:
        # detect objects
        # track objects
        results = model.track(frame, persist=True)

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

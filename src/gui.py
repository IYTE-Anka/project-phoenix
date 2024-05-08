from ultralytics_track import track_plot

import os
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

  source_path = os.path.join(os.path.dirname(__file__), "testing", args.source)
  model_path = os.path.join(os.path.dirname(__file__), "models", args.model)
  conf_threshold = float(args.conf)

  track_plot.run(source_path, model_path, conf_threshold, args.color)

if __name__ == "__main__":
  main()
import time
from picamera2 import Picamera2

picam2 = Picamera2()
config = picam2.create_still_configuration(lores={'size': (640, 480)}, display='lores', buffer_count=3)
picam2.configure(config)
picam2.start(show_preview=True)

time.sleep(15)  # wait until you want to do a capture, then:
picam2.capture_file("test.jpg")

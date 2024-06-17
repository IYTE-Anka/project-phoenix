import socket
import struct
from picamera2 import Picamera2, Preview
import cv2

# Initialize the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Set up the server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 8000))  # Host IP and port
server_socket.listen(0)

# Accept a single connection and make a file-like object out of it
connection, client_address = server_socket.accept()
connection = connection.makefile('wb')

try:
    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        size = len(data)
        connection.write(struct.pack('<L', size))
        connection.write(data)
finally:
    connection.close()
    server_socket.close()
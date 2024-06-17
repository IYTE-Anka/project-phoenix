import cv2
import socket
import numpy as np
import struct
import io

# Replace with your Raspberry Pi's IP address
HOST = '192.168.1.22'
PORT = 8000

# Create a socket to receive the video
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
connection = client_socket.makefile('rb')

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        # Display the image
        cv2.imshow('Video', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    connection.close()
    client_socket.close()
    cv2.destroyAllWindows()
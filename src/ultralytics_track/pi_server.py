import socket
import struct
from picamera2 import Picamera2, Preview
import cv2
import threading

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
connection_file = connection.makefile('wb')

def handle_incoming_data(conn):
    while True:
        try:
            print("Waiting for data size...")
            data_size_bytes = conn.recv(4)
            if len(data_size_bytes) < 4:
                continue
            
            data_size = struct.unpack('<L', data_size_bytes)[0]
            print(f"Data size: {data_size}")

            if data_size == 0:
                continue
            
            print("Waiting for data...")
            data = conn.recv(data_size)
            print("Data received:", data.decode('utf-8'))
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

incoming_thread = threading.Thread(target=handle_incoming_data, args=(connection,))
incoming_thread.start()

try:
    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        size = len(data)
        connection_file.write(struct.pack('<L', size))
        connection_file.write(data)
finally:
    connection.close()
    server_socket.close()

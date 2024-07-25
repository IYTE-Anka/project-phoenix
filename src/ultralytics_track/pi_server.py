import socket
import struct
from picamera2 import Picamera2, Preview
import cv2
import threading
import RPi.GPIO as gpio
import queue
from time import sleep

direction_pin_y   = 23
pulse_pin_y       = 24
direction_pin_x  = 17
pulse_pin_x      = 27

cw_direction    = 0 
ccw_direction   = 1 

gpio.setmode(gpio.BCM)
gpio.setup(direction_pin_y, gpio.OUT)
gpio.setup(pulse_pin_y, gpio.OUT)
gpio.setup(direction_pin_x, gpio.OUT)
gpio.setup(pulse_pin_x, gpio.OUT)
gpio.output(direction_pin_y,cw_direction)
gpio.output(direction_pin_x,cw_direction)

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

def handle_incoming_data(conn, data_queue):
    while True:
        try:
            data_size_bytes = conn.recv(4)
            if len(data_size_bytes) < 4:
                continue
    
            data_size = struct.unpack('<L', data_size_bytes)[0]
            if data_size == 0:
                continue
        
            data = conn.recv(data_size)
            data_queue.put(data.decode('utf-8'))
        except Exception as e:
            print(f"Error receiving data: {e}")
            break

data_queue = queue.Queue()

incoming_thread = threading.Thread(target=handle_incoming_data, args=(connection, data_queue))
incoming_thread.start()

try:
    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        size = len(data)
        connection_file.write(struct.pack('<L', size))
        connection_file.write(data)

        incoming_data = ""

        if not data_queue.empty():
            incoming_data = data_queue.get()
            print(f"INCOMING DATA: {incoming_data}")
        
        # SECTION: MOTOR CONTROL
        if incoming_data == "MLeft":
            gpio.output(direction_pin_x,cw_direction)
            for x in range(50):
                gpio.output(pulse_pin_x,gpio.HIGH)
                sleep(.001)
                gpio.output(pulse_pin_x,gpio.LOW)
                sleep(.0005)
        elif incoming_data == "MRight":
            gpio.output(direction_pin_x,ccw_direction)
            for x in range(50):
                gpio.output(pulse_pin_x,gpio.HIGH)
                sleep(.001)
                gpio.output(pulse_pin_x,gpio.LOW)
                sleep(.0005)
        elif incoming_data == "MUp":
            gpio.output(direction_pin_y,cw_direction)
            for x in range(50):
                gpio.output(pulse_pin_y,gpio.HIGH)
                sleep(.001)
                gpio.output(pulse_pin_y,gpio.LOW)
                sleep(.0005)
        elif incoming_data == "MDown":
            gpio.output(direction_pin_y,ccw_direction)
            for x in range(50):
                gpio.output(pulse_pin_y,gpio.HIGH)
                sleep(.001)
                gpio.output(pulse_pin_y,gpio.LOW)
                sleep(.0005)
finally:
    connection.close()
    server_socket.close()

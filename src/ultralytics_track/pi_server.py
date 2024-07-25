import socket
import struct
from picamera2 import Picamera2, Preview # type: ignore
import cv2
import threading
import RPi.GPIO as gpio # type: ignore
import queue
from time import sleep

# Define GPIO pins
direction_pin_y   = 23
pulse_pin_y       = 24
direction_pin_x  = 17
pulse_pin_x      = 27
direction_pin_gun = 5
pulse_pin_gun = 6

# Define directions
cw_direction    = 0 
ccw_direction   = 1 

# # Define parameters
# total_steps = 50  # Total number of steps
# ramp_steps = 15    # Number of steps for ramp-up and ramp-down
# constant_speed_steps = total_steps - 2 * ramp_steps  # Steps at constant speed

# # Define delay parameters
# initial_delay = 0.05/4  # Initial delay for ramp-up
# final_delay = 0.05/4  # Final delay for ramp-down
# constant_delay = 0.05/4  # Delay during constant speed


gpio.setmode(gpio.BCM)
gpio.setup(direction_pin_y, gpio.OUT)
gpio.setup(pulse_pin_y, gpio.OUT)
gpio.setup(direction_pin_x, gpio.OUT)
gpio.setup(pulse_pin_x, gpio.OUT)
gpio.setup(direction_pin_gun, gpio.OUT)
gpio.setup(pulse_pin_gun, gpio.OUT)
gpio.output(direction_pin_y,cw_direction)
gpio.output(direction_pin_x,cw_direction)
gpio.output(direction_pin_gun,cw_direction)

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
        
        # Function to control motor with ramp-up, constant speed, and ramp-down
        def control_motor(direction_pin, pulse_pin, direction, steps, speed):
            gpio.output(direction_pin, direction)
            
            for _ in range(steps):
                gpio.output(pulse_pin, gpio.HIGH)
                sleep(speed)
                gpio.output(pulse_pin, gpio.LOW)
                sleep(speed)

        # SECTION: MOTOR CONTROL
        if incoming_data == "MLeft":
            control_motor(direction_pin_x, pulse_pin_x, cw_direction, 50, 0.05/4)
        elif incoming_data == "MRight":
            control_motor(direction_pin_x, pulse_pin_x, ccw_direction, 50, 0.05/4)
        elif incoming_data == "MUp":
            control_motor(direction_pin_y, pulse_pin_y, cw_direction, 50, 0.05/4)
        elif incoming_data == "MDown":
            control_motor(direction_pin_y, pulse_pin_y, ccw_direction, 50, 0.05/4)
        elif incoming_data == "Ates":
            control_motor(direction_pin_gun, pulse_pin_gun, cw_direction, 200, 0.05/15)
finally:
    connection.close()
    server_socket.close()

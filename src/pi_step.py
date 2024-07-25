from time import sleep
import RPi.GPIO as gpio

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

try:
    while True:
        choice = input("Direction: ")
        if choice == "CW":
            sleep(.5)
            ax = input("Axis: ")
            if ax == "X":
                gpio.output(direction_pin_x,cw_direction)
                for x in range(50):
                    gpio.output(pulse_pin_x,gpio.HIGH)
                    sleep(.001)
                    gpio.output(pulse_pin_x,gpio.LOW)
                    sleep(.0005)
            elif ax == "Y":
                gpio.output(direction_pin_y,cw_direction)
                for x in range(50):
                    gpio.output(pulse_pin_y,gpio.HIGH)
                    sleep(.001)
                    gpio.output(pulse_pin_y,gpio.LOW)
                    sleep(.0005)

        elif choice == "CCW":
            sleep(.5)
            ax = input("Axis: ")
            if ax == "X":
                gpio.output(direction_pin_x,ccw_direction)
                for x in range(50):
                    gpio.output(pulse_pin_x,gpio.HIGH)
                    sleep(.001)
                    gpio.output(pulse_pin_x,gpio.LOW)
                    sleep(.0005)
            elif ax == "Y":
                gpio.output(direction_pin_y,ccw_direction)
                for x in range(50):
                    gpio.output(pulse_pin_y,gpio.HIGH)
                    sleep(.001)
                    gpio.output(pulse_pin_y,gpio.LOW)
                    sleep(.0005)
                

except KeyboardInterrupt:
    gpio.cleanup()

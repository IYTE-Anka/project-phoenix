from time import sleep
import RPi.GPIO as gpio

direction_pin_y   = 23
pulse_pin_y       = 24

cw_direction    = 0 
ccw_direction   = 1 

gpio.setmode(gpio.BCM)
gpio.setup(direction_pin_y, gpio.OUT)
gpio.setup(pulse_pin_y, gpio.OUT)
gpio.output(direction_pin_y,cw_direction)

try:
    while True:
        print('Direction CW')
        sleep(.5)
        gpio.output(direction_pin_y,cw_direction)
        for x in range(5000):
            gpio.output(pulse_pin_y,gpio.HIGH)
            sleep(.001)
            gpio.output(pulse_pin_y,gpio.LOW)
            sleep(.0005)

        print('Direction CCW')
        sleep(.5)
        gpio.output(direction_pin_y,ccw_direction)
        for x in range(5000):
            gpio.output(pulse_pin_y,gpio.HIGH)
            sleep(.001)
            gpio.output(pulse_pin_y,gpio.LOW)
            sleep(.0005)

except KeyboardInterrupt:
    gpio.cleanup()

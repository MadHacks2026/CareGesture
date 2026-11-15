from modulino import ModulinoMovement
from time import sleep_ms
import math

movement = ModulinoMovement()
THRESHOLD = 0.15 # adjust parameter

while True:
    ax, ay, az = movement.accelerometer
    gx, gy, gz = movement.gyro
    
    # Calculate total acceleration vector magnitude
    curr_mag = math.sqrt(ax**2 + ay**2 + az**2)
    delta = abs(curr_mag - prev_mag)
    
    if delta > THRESHOLD:
        print(f"Motion detected! Delta: {delta:.3f}g | Gyro: ({gx:.1f}, {gy:.1f}, {gz:.1f})")
        
    prev_mag = curr_mag
    sleep_ms(50)
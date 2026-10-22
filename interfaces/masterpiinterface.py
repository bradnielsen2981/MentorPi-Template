# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys, os, cv2, time, queue, logging, threading, math
sys.path.append('/home/pi/MasterPi')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import MLX90614
import numpy as np
import HiwonderSDK.Sonar as sonar
import HiwonderSDK.mecanum as mecanum
import HiwonderSDK.Board as Board
import HiwonderSDK.ActionGroupControl as actiongroup

from loggerinterface import setup_logger

# Define colors safely using a dictionary instead of variables and eval()
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255)
}

class MasterPiInterface():

    def __init__(self):
        """
        Initializes the low-level hardware connections to the Hiwonder board, 
        including sonar, mecanum chassis, infrared, and sets up logging.
        """
        self.logger = logging.getLogger('Robot')
        setup_logger(self.logger, '../logs/robot.log')

        self.sonar = sonar.Sonar()
        self.sonar.setRGBMode(0)
        self.chassis = mecanum.MecanumChassis()
        
        # Attempt to load the infrared sensor safely (Graceful Degradation)
        try:
            self.infrared = MLX90614.MLX90614()
            self.has_infrared = True
        except Exception as e:
            self.infrared = None
            self.has_infrared = False
            self.logger.warning(f"Infrared sensor not found during init: {e}")
        
        time.sleep(0.5)
        self.status = "Ready"
        self.arm_rotation = 1500 # centre position
        self.camera_pos = "default"

    def set_buzzer_time(self, timelimit=1):
        """Activates the board buzzer for a specified duration."""
        endtime = time.time() + timelimit
        Board.setBuzzer(1)
        while time.time() < endtime:
            time.sleep(0.01) # CPU Saver!
        Board.setBuzzer(0)

    def get_sonar_distance(self):
        """Reads and returns the current distance from the ultrasonic sensor."""
        distance = self.sonar.getDistance()
        time.sleep(0.1)
        return distance
    
    def get_infra_ambient(self):
        """Reads the ambient environmental temperature safely."""
        if self.has_infrared:
            try:
                return round(self.infrared.get_amb_temp(), 2)
            except Exception as e:
                # If the read fails, log it and turn off the sensor flag to prevent future lag/crashes
                self.logger.warning(f"Infrared ambient read failed: {e}")
                self.has_infrared = False 
                return 0.0
        return 0.0
    
    def get_infra_object(self):
        """Reads the object temperature safely."""
        if self.has_infrared:
            try:
                return round(self.infrared.get_obj_temp(), 2)
            except Exception as e:
                self.logger.warning(f"Infrared object read failed: {e}")
                self.has_infrared = False
                return 0.0
        return 0.0

    def set_sonarLED_colortuple(self, rgbtuple=(255, 0, 0)):
        """Sets the sonar 'eyes' using a direct RGB tuple."""
        self.sonar.setPixelColor(0, Board.PixelColor(*rgbtuple))
        self.sonar.setPixelColor(1, Board.PixelColor(*rgbtuple))
    
    def set_sonarLED_color(self, colour="black"):
        """Sets the sonar 'eyes' using a predefined color string."""
        rgb = COLORS.get(colour.lower(), COLORS["black"])
        self.sonar.setPixelColor(0, Board.PixelColor(*rgb))
        self.sonar.setPixelColor(1, Board.PixelColor(*rgb))
    
    def set_boardLED_colortuple(self, rgbtuple=(0, 0, 0)):
        """Sets the main board LEDs using a direct RGB tuple."""
        Board.RGB.setPixelColor(0, Board.PixelColor(*rgbtuple))
        Board.RGB.setPixelColor(1, Board.PixelColor(*rgbtuple))
        Board.RGB.show()
    
    def set_boardLED_color(self, colour="black"):        
        """Sets the main board LEDs using a predefined color string."""
        rgb = COLORS.get(colour.lower(), COLORS["black"])
        Board.RGB.setPixelColor(0, Board.PixelColor(*rgb))
        Board.RGB.setPixelColor(1, Board.PixelColor(*rgb))
        Board.RGB.show()
    
    def rotate_speed_time(self, rotationspeed=0.1, timelimit=3):
        """
        Rotates the robot chassis for a specific duration.
        Args:
            rotationspeed (float): Negative for anti-clockwise, positive for clockwise.
            timelimit (int): Time in seconds to execute the rotation.
        """
        self.status = "Rotating"
        self.chassis.set_velocity(24, 90, rotationspeed) 
        endtime = time.time() + timelimit
        
        while (time.time() < endtime) and (self.status == "Rotating"):
            time.sleep(0.01) # CPU Saver
            
        self.stop()
    
    def rotate_speed(self, rotationspeed=0.1):
        """Rotate forever - DANGEROUS AS THERE IS NO TIMELIMIT."""
        self.status = "Rotating"
        self.chassis.set_velocity(24, 90, rotationspeed) 
    
    def move_direction_time(self, power=35, direction=90, rotationspeed=0, timelimit=5):
        """
        Moves the chassis in a specific direction for a specific time.
        Args:
            power (int): Speed of the motors (capped at 40).
            direction (int): 90 is forward, 0 is right, 180 is left.
            rotationspeed (float): Rotational drift while moving.
            timelimit (int): Time in seconds to execute.
        """
        if power > 40:
            power = 40
            
        self.status = "Moving"
        self.chassis.set_velocity(power, direction, rotationspeed) 
        endtime = time.time() + timelimit
        
        while (time.time() < endtime) and (self.status == "Moving"):
            time.sleep(0.01) # CPU Saver
            
        self.stop()
    
    def move_direction(self, power=35, direction=90, rotationspeed=0):
        """Move in the direction indefinitely - DANGEROUS AS THERE IS NO TIMELIMIT."""
        if power > 40:
            power = 40
        self.status = "Moving"
        self.chassis.set_velocity(power, direction, rotationspeed) 
        
    def slide_direction(self, power=50, direction="left"):
        """Drifts sideways using the mecanum wheels indefinitely."""
        if power > 50:
            power = 50
            
        self.status = 'Drifting'
        dir_val = 180 if direction == 'left' else 0
        rot_speed = 0.3 if direction == 'left' else -0.3
        
        self.chassis.set_velocity(power, dir_val, rot_speed)  
    
    def slide_direction_time(self, power=50, direction="left", timelimit=2):
        """Drifts sideways using the mecanum wheels for a set time."""
        if power > 50:
            power = 50
            
        self.status = 'Drifting'
        dir_val = 180 if direction == 'left' else 0
        rot_speed = 0.3 if direction == 'left' else -0.3
            
        endtime = time.time() + timelimit
        self.chassis.set_velocity(power, dir_val, rot_speed)
        
        while (time.time() < endtime) and (self.status == 'Drifting'):
            time.sleep(0.01) # CPU Saver
            
        self.stop()    
    
    def get_status(self):
        """Returns the current movement status of the chassis."""
        return self.status

    def stop(self):
        """Instantly stops all wheel movement and resets status."""
        self.status = "Stopped"
        self.chassis.set_velocity(0, 0, 0)
    
    def run_arm_action(self, actionname="default"):
        """Runs a predefined action group file for the servos."""
        actiongroup.runAction(actionname)
    
    def reset_arm(self):
        """Resets the arm to the default folded posture."""
        actiongroup.runAction("default")
        self.camera_pos = "default"
    
    def get_voltage(self):
        """Returns the current battery voltage in Volts."""
        return Board.getBattery() / 1000.0
    
    def look_down(self):
        """Executes the macro to point the camera down at the floor."""
        actiongroup.runAction("lookdown")
        self.camera_pos = "lookdown"
    
    def look_up(self):
        """Executes the macro to point the camera up/forward."""
        actiongroup.runAction("lookup")
        self.camera_pos = "lookup"
    
    def stop_current_arm_action(self):
        """Force stops any currently executing action group."""
        actiongroup.stop_action_group()
    
    def rotate_arm_to_left_extreme(self):
        """Swings the entire robotic arm to the far left position."""
        Board.setPWMServoPulse(6, 2500, 1500)
        time.sleep(1.5)
        self.arm_rotation = 2500
    
    def grab_with_current_arm_rotation(self, deltaY=40): 
        """
        Executes a complex pickup sequence, dynamically adjusting the arm's 
        reach based on the target's Y-axis distance from the camera.
        """
        Board.setPWMServoPulse(1, 1864, 1000)
        time.sleep(1)
        Board.setPWMServoPulse(3, 900 + int(deltaY / 1.5), 2000) 
        time.sleep(2)
        Board.setPWMServoPulse(4, 2500 - int(deltaY * 1.7), 2000)  
        time.sleep(2)
        Board.setPWMServoPulse(5, 2050 + int(deltaY * 1.25), 2000) 
        time.sleep(2)
        Board.setPWMServoPulse(1, 1600, 2000) 
        time.sleep(2)

    def put_down_object(self):
        """Executes the macro to drop the currently held object."""
        self.run_arm_action("putdown")

    def rotate_arm(self, rotation=100):
        """
        Nudges the base arm servo by a specific degree amount.
        Constrains movement bounds to prevent mechanical damage.
        """
        if abs(rotation) <= 5:  # make the degrees either slow i.e 1 degree at a time
            rotation = int(math.copysign(1, rotation)) 
        elif abs(rotation) >= 100: # or cap the speed at 100 or -100
            rotation = int(math.copysign(100, rotation))
            
        self.arm_rotation = self.arm_rotation + rotation
                
        if self.arm_rotation < 500:
            self.arm_rotation = 500
        elif self.arm_rotation > 2500:
            self.arm_rotation = 2500
        else:
            dtime = abs(rotation) * 10        
            Board.setPWMServoPulse(6, self.arm_rotation, dtime)
            time.sleep(dtime / 1000.0)

# main execution point for testing purposes
if __name__ == '__main__':
    ROBOT = MasterPiInterface()
    print("\033c")
    time.sleep(3)
    
    v = ROBOT.get_voltage()
    print(f"Voltage: {v}V")
    
    input("Press Enter to Slide")
    ROBOT.slide_direction_time()
    ROBOT.stop()
    
    input("Buzzer on")
    ROBOT.set_buzzer_time(1)
    
    input("Sonar purple")
    ROBOT.set_sonarLED_color("purple")
    
    input("Board red")
    ROBOT.set_boardLED_color("red")
    
    print("SONAR", ROBOT.get_sonar_distance())
    print("INFRARED_AMBIENT_TEMP", ROBOT.get_infra_ambient())
    print("INFRARED OBJECT TEMP", ROBOT.get_infra_object())
    
    ROBOT.look_down()
    input("Move forward press enter")
    ROBOT.move_direction_time(timelimit=2)
    
    input("Move left press enter")
    ROBOT.move_direction_time(power=35, direction=0, rotationspeed=0, timelimit=2)
    
    input("Move right press enter")
    ROBOT.move_direction_time(power=35, direction=180, rotationspeed=0, timelimit=2)
    
    input("Move back press enter")
    ROBOT.move_direction_time(power=35, direction=270, rotationspeed=0, timelimit=2)
    
    input("Rotate left - press enter")
    ROBOT.rotate_speed_time(rotationspeed=-0.1, timelimit=2)
    
    input("Rotate right - press enter")
    ROBOT.rotate_speed_time(rotationspeed=0.1, timelimit=2)
    
    input("Robot grab with current arm rotation")
    ROBOT.grab_with_current_arm_rotation()
    
    time.sleep(1)
    ROBOT.reset_arm()
    ROBOT.set_sonarLED_color("black")
    ROBOT.set_boardLED_color("black")
    ROBOT.stop()
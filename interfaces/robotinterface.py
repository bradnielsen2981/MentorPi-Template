# -*- coding: utf-8 -*-
# container for more advanced functionality
# use Ctrl + K, Ctrl + 0 to minimise all coding blocks in VSCODE
import time, sys, os, math, logging
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from masterpiinterface import MasterPiInterface
from soundinterface import SoundInterface
from camerainterface import CameraInterface

class RobotInterface(MasterPiInterface):
     
    def __init__(self):
        """
        Initializes the Robot Interface, sets up sound and camera modules, 
        and waits for hardware to warm up.
        """
        self.command = "Ready" # keep track of user commands
        self.show_camera = False # only turn this on to see the camera window, will slow down performance when using VNC
        self.SOUND = SoundInterface()
        
        self.CAMERA = CameraInterface()
        self.CAMERA.start()
        time.sleep(2) # Camera warmup time since it runs in a background thread now
        
        super().__init__()
        self.starttime = time.time() # when did robot get created
        return
    
    def stop_command(self):
        """
        Safely stops the robot's current movement process and resets the command state.
        Call this anytime you need to abort a running while-loop instantly.
        """
        self.command = "Ready"
        self.stop()
        return

    def move_toward_colour_detected(self, colour="red", timelimit=5, mode='turning'):
        """
        Instructs the robot to continuously track and move towards a specific color blob.
        Calculates the angle of the blob relative to the camera center and adjusts trajectory.
        
        Args:
            colour (str): The color name from your YAML file to track. Defaults to "red".
            timelimit (int/float): Maximum time in seconds to execute the command before giving up. Defaults to 5.
            mode (str): Movement style. 
                        - 'turning' will pivot the robot's chassis to face the object.
                        - 'drifting' will use mecanum wheels to slide diagonally toward it.
                        
        Returns:
            dict: The final detection data snapshot and timestamp.
        """
        self.command = "move_toward_colour_detected"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        
        # Give visual feedback that tracking has started
        self.set_boardLED_color(colour)
        self.CAMERA.add_detection_task("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        endtime = time.time() + timelimit
        
        # Determine how close the robot needs to get based on where the camera is looking
        distance = 145
        if self.camera_pos == "lookdown":
            distance = 250
        elif self.camera_pos == "lookup":
            distance = 90 
        elif self.camera_pos == "default":
            distance = 145    
            
        while (time.time() < endtime) and (self.command == "move_toward_colour_detected"):
            if not self.show_camera_window(): # Acts as our loop throttler
                break
            
            data.update(self.CAMERA.get_detection_data())
            
            colour_data = data.get('detect_colour', {}).get(colour, {})
            if colour_data.get('found'):
                # Extract the bounding box details
                center, size, angle = colour_data['rect']
                x, y = center
                
                # Calculate offsets from the exact center of the 640x480 frame
                deltaY = 480 - y
                deltaX = 320 - x
                
                # Calculate the exact angle to the object using trigonometry
                theta_degrees = int(math.degrees(math.atan2(deltaX, deltaY)) + 90)
                
                # If we are not close enough yet, keep moving
                if deltaY > distance:
                    if mode == 'drifting':
                        self.move_direction(power=33, direction=theta_degrees, rotationspeed=0)
                    elif mode == 'turning':
                        turn = round(math.radians(theta_degrees - 90) / 8, 2)
                        # Deadzone to prevent micro-stuttering if it's already basically centered
                        if abs(turn) <= 0.01:
                            turn = 0
                        self.move_direction(power=33, direction=90, rotationspeed=-turn)
                else:
                    # We have arrived at the object! Do a tiny backup to brake.
                    self.move_direction_time(direction=270, rotationspeed=0, power=33, timelimit=0.2)
                    break
                
        self.stop_command()        
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        return data

    def move_direction_until_detection(self, movetype='forward', distanceto=250, detection_types=None,
                                       detection_colours=None, timelimit=5, confirmlevel=1, target_class='person'):
        """
        Moves the robot in a continuous pattern until a specific sensor or visual target is triggered.
        Highly versatile function for autonomous exploration.
        
        Args:
            movetype (str): How to move ('forward', 'turnright', 'turnleft', 'circleright', 'circleleft', 'slideright', 'slideleft').
            distanceto (int): Threshold distance for Sonar (cm) or Line (Y-pixel coordinate). Defaults to 250.
            detection_types (list): What to look for (['sonar', 'line', 'colour', 'model', 'all']). 
            detection_colours (list): Which specific colors to look for if 'colour' is active. Defaults to ['red'].
            timelimit (int): Max time to spend searching before giving up. Defaults to 5.
            confirmlevel (int): How many distinct detection types must be triggered simultaneously to succeed. Defaults to 1.
            target_class (str): Which specific EdgeTPU model class to trigger on (e.g., 'person', 'apple').
            
        Returns:
            dict: The final detection data snapshot and timestamp.
        """
        if detection_types is None:
            detection_types = ['colour']
        if detection_colours is None:
            detection_colours = ['red']
            
        self.command = "move_direction_until_detection"
        data = {'command': self.command, 'movetype': movetype }
        detections = [] # Keeps track of which sensors have tripped
        
        # 1. Setup the camera tasks based on user arguments
        if 'all' in detection_types:
            self.CAMERA.detect_all()
        else:            
            if 'line' in detection_types:
                self.CAMERA.add_detection_task("detect_line")
            if 'colour' in detection_types:
                self.CAMERA.add_detection_task('detect_colour')
                self.CAMERA.set_detection_colours(detection_colours)
            if 'model' in detection_types:
                self.CAMERA.load_detection_model()
                self.CAMERA.add_detection_task("detect_model")
          
        data['starttime'] = time.time()
        endtime = time.time() + timelimit
        
        # 2. Initiate the continuous movement pattern
        if movetype:
            if movetype == "forward":
                self.move_direction(power=33, direction=90, rotationspeed=0) 
            elif movetype == "turnright":
                self.rotate_speed(rotationspeed=0.08)
            elif movetype == "turnleft":
                self.rotate_speed(rotationspeed=-0.08)
            elif movetype == "circleright":
                self.move_direction(power=35, direction=90, rotationspeed=0.08)
            elif movetype == "circleleft":
                self.move_direction(power=35, direction=90, rotationspeed=-0.08)
            elif movetype == "slideright":
                self.move_direction(power=40, direction=0, rotationspeed=0)
            elif movetype == "slideleft":
                self.move_direction(power=40, direction=-180, rotationspeed=0)    
        
        # 3. Enter the active scanning loop
        while (time.time() < endtime) and (self.command == "move_direction_until_detection"):
            num_detections_processed = 0
            if not self.show_camera_window():
                break
            
            data.update(self.CAMERA.get_detection_data())
            
            # --- SONAR CHECK ---
            if 'sonar' in detection_types or 'all' in detection_types:
                sonar_distance = self.get_sonar_distance()
                data['detect_sonar'] = { 'distance': sonar_distance }
                self.CAMERA.output_message += f" Sonar: {sonar_distance}"
                
                # Check 0 < sonar_distance to prevent false positives from sensor read errors
                if 0 < sonar_distance < distanceto:
                    print("Sonar detected!")
                    if 'sonar' not in detections:
                        detections.append('sonar')
                        if len(detections) == confirmlevel:
                            break # Success! Break the while loop
                
                num_detections_processed += 1
                if num_detections_processed == len(detection_types):
                    continue
            
            # --- LINE CHECK ---
            if 'line' in detection_types or 'all' in detection_types:
                line_data = data.get('detect_line', {})
                if line_data.get('found'):
                    line = line_data['line']
                    cx = (line[0][0] + line[1][0]) / 2
                    cy = (line[0][1] + line[1][1]) / 2
                    
                    if (480 - cy < distanceto):
                        angle_rad = np.arctan2(line[1][1] - line[0][1], line[1][0] - line[0][0])
                        angle = np.degrees(angle_rad)
                        if -45 < angle < 45: # somewhat horizontal lines
                            print("Line detected!")
                            if 'line' not in detections:
                                detections.append('line')
                                if len(detections) == confirmlevel:
                                    break
                                
                num_detections_processed += 1
                if num_detections_processed == len(detection_types):
                    continue
                
            # --- COLOUR CHECK ---
            if 'colour' in detection_types or 'all' in detection_types:
                for col in data.get('detect_colour', {}).keys():
                    col_data = data['detect_colour'][col]
                    if col_data.get('found'):
                        print(f"{col} detected!")
                        if col not in detections:
                            detections.append(col)
                            if len(detections) == confirmlevel:
                                break
                
                if len(detections) == confirmlevel:
                    break
                
                num_detections_processed += 1
                if num_detections_processed == len(detection_types):
                    continue
                
            # --- NEURAL NETWORK CHECK ---
            if 'model' in detection_types or 'all' in detection_types: 
                model_data = data.get('detect_model', {})
                if model_data.get('found'):
                    for target in model_data['targets']:
                        if target['class'] == target_class:
                            if 'model' not in detections:
                                print(f"{target_class} detected!")
                                detections.append('model')
                                if len(detections) == confirmlevel:
                                    break
                                    
                    if len(detections) == confirmlevel:
                        break

        self.stop_command()
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        return data

    def rotate_arm_until_colour_detected(self, colour="red", timelimit=10):
        """
        Rotates the mechanical arm from left to right until the specified color enters the frame.
        Useful for visually sweeping an area without moving the chassis.
        
        Args:
            colour (str): Target color name from YAML. Defaults to "red".
            timelimit (int): Max search time in seconds. Defaults to 10.
            
        Returns:
            dict: Data dictionary including the final 'arm_rotation' PWM value where it stopped.
        """
        self.command = "rotate_arm_until_colour_detected"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        
        self.rotate_arm_to_left_extreme()
        self.CAMERA.add_detection_task("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        
        endtime = time.time() + timelimit
        while (time.time() < endtime) and (self.command == "rotate_arm_until_colour_detected"):
            if not self.show_camera_window():
                break
            
            data.update(self.CAMERA.get_detection_data())
            
            colour_data = data.get('detect_colour', {}).get(colour, {})
            if colour_data.get('found'):
                center, size, angle = colour_data['rect']
                x, y = center
            elif 'detect_colour' in data: # Only rotate if we've processed a frame
                self.rotate_arm(-100)
                    
        self.stop_command()
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        data['arm_rotation'] = self.arm_rotation
        return data

    def rotate_robot_to_arm_rotation(self, timelimit=5):
        """
        TO DO - Rotate robot chassis until robot is aligned with the current arm_rotation.
        
        Args:
            timelimit (int): Max alignment time in seconds. Defaults to 5.
        """
        self.command = "rotate_robot_to_arm_rotation"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        self.stop_command()
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        return data
    
    def rotate_arm_until_colour_detected_is_centered(self, colour="red", timelimit=10):
        """
        Rotates the arm minutely to perfectly center the target color in the X-axis of the camera frame.
        Usually executed right before attempting a physical pickup.
        
        Args:
            colour (str): Target color name from YAML. Defaults to "red".
            timelimit (int): Max alignment time in seconds. Defaults to 10.
            
        Returns:
            dict: Includes the final 'x' and 'y' coordinates of the centered object.
        """
        self.look_down() # must be in look down mode to line up a grab
        self.command = "rotate_arm_until_colour_detected_is_centered"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        
        self.CAMERA.add_detection_task("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        self.set_boardLED_color(colour)
        
        centered = False
        x, y = 0, 0
        endtime = time.time() + timelimit
        
        while (time.time() < endtime) and (self.command == "rotate_arm_until_colour_detected_is_centered"):
            if not self.show_camera_window():
                break
            
            data.update(self.CAMERA.get_detection_data())
            
            colour_data = data.get('detect_colour', {}).get(colour, {})
            if colour_data.get('found') and not centered:
                center, size, angle = colour_data['rect']
                x, y = center
                deltaX = 320 - x # How far off from center is it?
                rotation = int(deltaX / 320 * 500) # Convert pixel offset to PWM rotation steps
                
                if abs(rotation) > 2:
                    self.rotate_arm(rotation) # Nudge it closer
                else:
                    centered = True # Object is dead center!
                    break
                
        self.stop_command()        
        self.CAMERA.end_detection()
        data['x'] = x
        data['y'] = y
        data['endtime'] = time.time()
        return data

    def pick_up_centered_object_with_look_down(self, y):
        """
        Executes the mechanical pickup sequence. Converts the object's Y-pixel coordinate 
        into a physical distance to calculate how far the arm needs to extend forward.
        
        Args:
            y (int/float): The Y-coordinate of the object in the 640x480 frame.
            
        Returns:
            dict: Contains a boolean 'pickup' indicating if the arm sequence triggered.
        """
        self.command = "pick_up_centered_object_with_look_down"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        
        # Ensure object is close enough to grab (bottom half of the screen)
        if y >= 100:
            deltaY = int((480 - y) / 240 * 300) # Convert pixels to PWM extension mapping
            self.grab_with_current_arm_rotation(deltaY)
            self.reset_arm()
            data['pickup'] = True
        else:
            data['pickup'] = False
            
        self.stop_command()
        data['endtime'] = time.time()
        return data
    
    def was_object_pickup_successful(self, colour='red', timelimit=10):
        """
        Visually validates if the target object is currently being held in the claw.
        It does this by checking if a very large mass of the target color is right at the 
        bottom edge of the camera view (where the claw sits).
        
        Args:
            colour (str): Target color name from YAML. Defaults to "red".
            timelimit (int): Max time to look for the held object. Defaults to 10.
            
        Returns:
            dict: Contains boolean 'success' indicating if the object is secure.
        """
        self.command = "was_object_pickup_successful"
        self.CAMERA.add_detection_task("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        
        data = {
            'command': self.command,
            'success': False,
            'starttime': time.time()
        }
        
        endtime = time.time() + timelimit
        while (time.time() < endtime) and (self.command == "was_object_pickup_successful"):
            if not self.show_camera_window():
                break
            
            data.update(self.CAMERA.get_detection_data())
            
            colour_data = data.get('detect_colour', {}).get(colour, {})
            if colour_data.get('found'):
                center, size, angle = colour_data['rect']
                width, height = size
                x, y = center
                deltaY = 480 - y
                
                # Is it at the bottom edge (<100px), large enough (250-500px width), and relatively straight?
                if (deltaY < 100) and (250 < width < 500) and (abs(angle) < 5):
                    data['success'] = True
                    break
                
        self.stop_command()
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        return data
 
    def show_camera_window(self):
        """
        Renders the OpenCV display window. 
        If self.show_camera is False, it implements a small artificial sleep. 
        This is a critical performance fix to prevent headless while-loops from 
        spinning out of control and maxing out the CPU.
        
        Returns:
            bool: True normally, False if the user presses 'q' on the keyboard.
        """
        if not self.show_camera:
            # CRITICAL FIX: Add a small delay to prevent while-loops from running unthrottled
            time.sleep(0.03) 
            return True
        
        frame = self.CAMERA.get_frame()
        cv2.imshow('Detection Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def auto_detection(self, timelimit=100000000): 
        """
        Runs the robot in a stationary mode, actively logging all visible targets.
        Essentially a sandbox mode for testing vision capabilities without movement.
        
        Args:
            timelimit (int): Time to run. Defaults to a very large number for infinite runtime.
        """
        self.command = "auto_detection"
        data = {
            'command': self.command,
            'starttime': time.time()
        }
        
        endtime = time.time() + timelimit
        self.CAMERA.detect_all()
        
        while (time.time() < endtime) and (data['command'] == "auto_detection"): 
            if not self.show_camera_window():
                break
            data.update(self.CAMERA.get_detection_data())
            
        self.stop_command()
        self.CAMERA.end_detection()
        data['endtime'] = time.time()
        self.CAMERA.detection_data_expire_time = 1
        return data

    def cycle_through_all_detection_tasks(self, duration_per_task=5):
        """
        Cycles through a baseline (no detection) and each detection task (line, colour, model) 
        individually. Used to profile the Raspberry Pi and observe how much overhead 
        each algorithm adds to the baseline stream.
        
        Args:
            duration_per_task (int): How long to run each task in seconds. Defaults to 5.
        """
        self.command = "cycle_through_all_detection_tasks"
        
        tasks_to_test = ['baseline', 'detect_line', 'detect_colour', 'detect_model']
        
        # Pre-load dependencies
        self.CAMERA.set_detection_colours(['red', 'green', 'blue'])
        self.CAMERA.load_detection_model()
        self.CAMERA.turn_on_output_text() 
        
        print("\n========================================")
        print(" STARTING DETECTION PERFORMANCE PROFILER")
        print("========================================\n")
        
        for task in tasks_to_test:
            self.CAMERA.clear_detection_tasks()
            self.CAMERA.clear_detection_data() 
            
            if task == 'baseline':
                print(f">>> Testing 'Baseline (No Detection)' for {duration_per_task} seconds...")
                self.SOUND.say("baseline test")
                self.CAMERA.set_output_message("BASELINE - RAW FEED")
            else:
                print(f">>> Testing '{task}' for {duration_per_task} seconds...")
                self.SOUND.say(task.replace('_', ' '))
                self.CAMERA.add_detection_task(task)
                self.CAMERA.set_output_message(task.upper())
            
            endtime = time.time() + duration_per_task
            
            while (time.time() < endtime) and (self.command == "cycle_through_all_detection_tasks"):
                if not self.show_camera_window():
                    self.command = "Stop" 
                    break
            
            if self.command != "cycle_through_all_detection_tasks":
                print("\n[!] Test aborted by user.")
                break 
                
        print("\n========================================")
        print(" CYCLE COMPLETE ")
        print("========================================\n")
        
        # Clean up
        self.CAMERA.set_output_message("")
        self.stop_command()
        self.CAMERA.end_detection()
        return

    def shutdown(self):
        """
        Safely powers down all systems. Stops motors, stops the background camera threads, 
        and clears board LEDs. Always call this when exiting the script to prevent hardware damage.
        """
        self.command = "Shutdown"
        self.stop_command()
        self.CAMERA.stop()
        self.set_sonarLED_color()
        self.set_boardLED_color()
        return

# TEST ROBOT CODE
if __name__ == '__main__':
    ROBOT = RobotInterface()
    
    try:
        ROBOT.stop()
        print("\033c")
        input("Press Enter to Start: ")
        
        ROBOT.CAMERA.create_detection_window()
        ROBOT.show_camera = True # THIS WILL SLOW DOWN THE FRAME RATE IF ON VNC
        print("Voltage: ", ROBOT.get_voltage())
        
        ROBOT.look_up()
        ROBOT.SOUND.say("Robot ready")

        # Test all detection tasks
        #ROBOT.cycle_through_all_detection_tasks(duration_per_task=20)

        # Uncomment below to test movements
        data = ROBOT.move_direction_until_detection(movetype='forward', distanceto=250, detection_types=['sonar','color'], confirmlevel=2, detection_colours=['red'], timelimit=10)
        print(data)

    except KeyboardInterrupt:
        # Failsafe if you cancel the script mid-movement!
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the robot stops moving and shuts down cleanly
        ROBOT.shutdown()
        cv2.destroyAllWindows()
        sys.exit(0)
# -*- coding: utf-8 -*-
# container for more advanced functionality
# use Ctrl + K, Ctrl + 0 to minimise all coding blocks in VSCODE
import time, sys, os, math, logging
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mentorpiinterface import MentorPi
from soundinterface import SoundInterface
from camerainterface import CameraInterface
from loggerinterface import setup_logger

# RGB values used to show a detection colour on the robot's LEDs
LED_COLOURS = {
    'red': (255, 0, 0), 'green': (0, 255, 0), 'dark_green': (0, 100, 0), 'blue': (0, 0, 255),
    'yellow': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
}

# Speeds used by the movement behaviours
DRIVE_SPEED = 0.15 # m/s
TURN_SPEED = 0.5   # rad/s

class RobotInterface(MentorPi):

    def __init__(self):
        """
        Initializes the Robot Interface, sets up sound and camera modules,
        and waits for hardware to warm up.
        """
        self.command = "Ready" # keep track of user commands
        self.show_camera = False # only turn this on to see the camera window, will slow down performance when using VNC
        self.SOUND = SoundInterface()

        self.logger = logging.getLogger('Robot')
        setup_logger(self.logger, '../logs/robot.log')

        super().__init__() # connects to the robot through ROS 2

        self.CAMERA = CameraInterface(node=self.node) # shares the robot's ROS connection
        self.CAMERA.start()
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

    def set_led_colour(self, colour=None):
        """
        Shows a colour on the robot's two RGB LEDs.

        Args:
            colour (str, optional): A colour name such as 'red' or 'blue'. None turns the LEDs off.
        """
        self.turn_on_led(*LED_COLOURS.get(colour, (0, 0, 0)))

    def move_toward_colour_detected(self, colour="red", timelimit=5, mode='turning'):
        """
        Instructs the robot to continuously track and move towards a specific color blob.
        Calculates the angle of the blob relative to the camera center and adjusts trajectory.

        Args:
            colour (str): The color name from your YAML file to track. Defaults to "red".
            timelimit (int/float): Maximum time in seconds to execute the command before giving up. Defaults to 5.
            mode (str): Movement style.
                        - 'turning' will turn the robot's chassis to face the object while driving forward.
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
        self.set_led_colour(colour)
        self.CAMERA.add_detection_task("detect_colour")
        self.CAMERA.add_detection_colour(colour)
        endtime = time.time() + timelimit

        # How close (in pixels from the bottom of the frame) the object must get before stopping
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

                # Angle to the object: 0 = straight ahead, positive = to the left
                angle_to_object = math.atan2(deltaX, deltaY)

                # If we are not close enough yet, keep moving
                if deltaY > distance:
                    if mode == 'drifting':
                        self.set_velocity(linear_x=DRIVE_SPEED * math.cos(angle_to_object),
                                          linear_y=DRIVE_SPEED * math.sin(angle_to_object))
                    elif mode == 'turning':
                        turn = angle_to_object
                        # Deadzone to prevent micro-stuttering if it's already basically centered
                        if abs(turn) <= 0.05:
                            turn = 0
                        self.set_velocity(linear_x=DRIVE_SPEED, angular_z=turn)
                else:
                    # We have arrived at the object!
                    break
            else:
                self.stop() # lost sight of it, wait for it to reappear

        self.stop_command()
        self.CAMERA.end_detection()
        self.set_led_colour()
        data['endtime'] = time.time()
        return data

    def move_direction_until_detection(self, movetype='forward', distanceto=250, detection_types=None,
                                       detection_colours=None, timelimit=5, confirmlevel=1, target_class='person'):
        """
        Moves the robot in a continuous pattern until a specific sensor or visual target is triggered.
        Highly versatile function for autonomous exploration.

        Args:
            movetype (str): How to move ('forward', 'turnright', 'turnleft', 'circleright', 'circleleft', 'slideright', 'slideleft').
            distanceto (int): Threshold distance for Lidar (cm) or Line (pixels from the bottom of the frame). Defaults to 250.
            detection_types (list): What to look for (['lidar', 'line', 'colour', 'model', 'all']).
            detection_colours (list): Which specific colors to look for if 'colour' is active. Defaults to ['red'].
            timelimit (int): Max time to spend searching before giving up. Defaults to 5.
            confirmlevel (int): How many distinct detection types must be triggered simultaneously to succeed. Defaults to 1.
            target_class (str): Which YOLOv5 object class to trigger on (e.g., 'person', 'cup').

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

        # 2. Choose the continuous movement pattern: (forward m/s, left m/s, anticlockwise rad/s)
        velocities = {
            "forward": (DRIVE_SPEED, 0, 0),
            "turnright": (0, 0, -TURN_SPEED),
            "turnleft": (0, 0, TURN_SPEED),
            "circleright": (DRIVE_SPEED, 0, -TURN_SPEED),
            "circleleft": (DRIVE_SPEED, 0, TURN_SPEED),
            "slideright": (0, -DRIVE_SPEED, 0),
            "slideleft": (0, DRIVE_SPEED, 0),
        }
        velocity = velocities.get(movetype, (0, 0, 0))

        # 3. Enter the active scanning loop
        while (time.time() < endtime) and (self.command == "move_direction_until_detection"):
            self.set_velocity(*velocity) # resent every loop so the robot keeps moving
            num_detections_processed = 0
            if not self.show_camera_window():
                break

            data.update(self.CAMERA.get_detection_data())

            # --- LIDAR CHECK ---
            if 'lidar' in detection_types or 'all' in detection_types:
                lidar_distance = self.get_distance_ahead() # metres, inf if nothing in range
                lidar_cm = None if lidar_distance is None else lidar_distance * 100
                data['detect_lidar'] = { 'distance': lidar_cm }
                self.CAMERA.output_message += f" Lidar: {lidar_cm}"

                if lidar_cm is not None and lidar_cm < distanceto:
                    print("Lidar detected!")
                    if 'lidar' not in detections:
                        detections.append('lidar')
                        if len(detections) == confirmlevel:
                            break # Success! Break the while loop

                num_detections_processed += 1
                if num_detections_processed == len(detection_types):
                    continue

            # --- LINE CHECK ---
            if 'line' in detection_types or 'all' in detection_types:
                # A crossing line is one running across the robot's path
                crossing = data.get('detect_line', {}).get('crossing')
                if crossing and (480 - crossing['y'] < distanceto):
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

        while (time.time() < endtime) and (self.command == "auto_detection"):
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
        Cycles through a baseline (no detection) and each detection task (line, colour, model, depth)
        individually. Used to profile the Raspberry Pi and observe how much overhead
        each algorithm adds to the baseline stream.

        Args:
            duration_per_task (int): How long to run each task in seconds. Defaults to 5.
        """
        self.command = "cycle_through_all_detection_tasks"

        tasks_to_test = ['baseline', 'detect_line', 'detect_colour', 'detect_model', 'detect_depth']

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
        and clears the LEDs. Always call this when exiting the script to prevent hardware damage.
        """
        self.command = "Shutdown"
        self.stop_command()
        self.CAMERA.stop()
        self.turn_off_led()
        super().shutdown() # releases the ROS connection
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
        print("Voltage: ", ROBOT.get_battery_voltage())

        ROBOT.center_camera()
        ROBOT.SOUND.say("Robot ready")

        # Test all detection tasks
        #ROBOT.cycle_through_all_detection_tasks(duration_per_task=20)

        # Uncomment below to test movements
        data = ROBOT.move_direction_until_detection(movetype='forward', distanceto=50, detection_types=['lidar','colour'], confirmlevel=2, detection_colours=['red'], timelimit=10)
        print(data)

    except KeyboardInterrupt:
        # Failsafe if you cancel the script mid-movement!
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the robot stops moving and shuts down cleanly
        ROBOT.shutdown()
        cv2.destroyAllWindows()
        sys.exit(0)

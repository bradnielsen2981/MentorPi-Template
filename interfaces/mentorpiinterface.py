#!/usr/bin/env python3
# encoding: utf-8
"""
mentorpiinterface.py - simple Python API for the Hiwonder MentorPi (mecanum).

Talks to the robot through the ROS 2 topics published by the stock bringup
(ros_robot_controller, odom_publisher, imu filter, MS200 lidar, ascamera), so
the bringup must already be running. Run scripts inside the MentorPi container
with the ROS environment sourced, e.g.

    docker exec -it -u ubuntu MentorPi zsh
    cd ~/shared/MentorPiTemplate && python3 my_script.py

Example:

    from interfaces.mentorpiinterface import MentorPi

    with MentorPi() as robot:
        robot.move_forward(2)                    # 2 s forward
        robot.move_sideways(1, 'left')           # 1 s strafe left
        robot.rotate(1.5, 'clockwise')
        print(robot.get_imu_sensor())
        robot.turn_on_led(0, 255, 0)
        robot.beep()
"""
import math
import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import UInt16
from ros_robot_controller_msgs.msg import (
    BuzzerState, ButtonState, LedState, MotorState, MotorsState, OLEDState,
    PWMServoState, RGBState, RGBStates, SetPWMServoState
)

# Topics provided by the stock MentorPi bringup.
CMD_VEL_TOPIC = '/controller/cmd_vel'
MOTOR_TOPIC = '/ros_robot_controller/set_motor'
RGB_TOPIC = '/ros_robot_controller/set_rgb'
LED_TOPIC = '/ros_robot_controller/set_led'
BUZZER_TOPIC = '/ros_robot_controller/set_buzzer'
OLED_TOPIC = '/ros_robot_controller/set_oled'
PWM_SERVO_TOPIC = '/ros_robot_controller/pwm_servo/set_state'
BATTERY_TOPIC = '/ros_robot_controller/battery'
BUTTON_TOPIC = '/ros_robot_controller/button'
IMU_TOPIC = '/imu'                      # calibrated + complementary filtered
IMU_RAW_TOPIC = '/ros_robot_controller/imu_raw'
IMU_RPY_TOPIC = '/imu/rpy/filtered'
ODOM_TOPIC = '/odom'                    # EKF fused odometry
SCAN_TOPIC = '/scan_raw'
RGB_IMAGE_TOPIC = '/ascamera/camera_publisher/rgb0/image'
DEPTH_IMAGE_TOPIC = '/ascamera/camera_publisher/depth0/image_raw'

# Default folder for save_camera_image / save_depth_image (<project>/static/photos).
PHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'static', 'photos')

# Camera gimbal PWM servos (pulse width in microseconds, 1500 = centre).
CAMERA_TILT_SERVO_ID = 1
CAMERA_PAN_SERVO_ID = 2
SERVO_CENTER = 1500
SERVO_MIN = 500
SERVO_MAX = 2500

# Safety limits for velocity commands.
MAX_LINEAR_SPEED = 0.4    # m/s
MAX_ANGULAR_SPEED = 3.0   # rad/s
CMD_RATE_HZ = 20.0        # re-publish rate while a timed move is running


def _photo_path(prefix, ext):
    """Timestamped file path in static/photos, e.g. camera_20261024_203015_123.jpg."""
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}'
    return os.path.join(PHOTO_DIR, f'{prefix}_{stamp}{ext}')


def _clamp(value, low, high):
    return max(low, min(high, value))


def _quaternion_to_rpy(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(_clamp(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


class MentorPi:
    """High level interface to the MentorPi robot.

    Movement functions block for the requested time and then stop the robot.
    Sensor getters return the most recent message received (or None if no
    data has arrived yet).
    """

    def __init__(self, node_name='mentorpi_interface', default_speed=0.2,
                 default_turn_speed=1.0):
        self.default_speed = default_speed
        self.default_turn_speed = default_turn_speed

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.node = Node(node_name)

        self._lock = threading.Lock()
        self._latest = {}
        self._events = {}
        self._button_callbacks = []

        # Publishers
        self._cmd_vel_pub = self.node.create_publisher(Twist, CMD_VEL_TOPIC, 1)
        self._motor_pub = self.node.create_publisher(MotorsState, MOTOR_TOPIC, 10)
        self._rgb_pub = self.node.create_publisher(RGBStates, RGB_TOPIC, 10)
        self._led_pub = self.node.create_publisher(LedState, LED_TOPIC, 10)
        self._buzzer_pub = self.node.create_publisher(BuzzerState, BUZZER_TOPIC, 10)
        self._oled_pub = self.node.create_publisher(OLEDState, OLED_TOPIC, 10)
        self._servo_pub = self.node.create_publisher(SetPWMServoState, PWM_SERVO_TOPIC, 10)

        # Subscribers: just cache the latest message of each.
        self._subscribe('imu', Imu, IMU_TOPIC)
        self._subscribe('imu_raw', Imu, IMU_RAW_TOPIC)
        self._subscribe('imu_rpy', Vector3Stamped, IMU_RPY_TOPIC)
        self._subscribe('odom', Odometry, ODOM_TOPIC)
        self._subscribe('scan', LaserScan, SCAN_TOPIC)
        self._subscribe('battery', UInt16, BATTERY_TOPIC, qos=10)
        self._subscribe('button', ButtonState, BUTTON_TOPIC, qos=10,
                        callback=self._on_button)
        # Camera topics are subscribed lazily since images are expensive.
        self._camera_subscribed = set()
        self._bridge = None

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Give DDS a moment to match publishers/subscribers.
        time.sleep(0.5)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def shutdown(self):
        """Stop the robot and release ROS resources."""
        try:
            self.stop()
            time.sleep(0.1)
        finally:
            self._executor.shutdown()
            self.node.destroy_node()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _subscribe(self, key, msg_type, topic, qos=qos_profile_sensor_data, callback=None):
        self._events[key] = threading.Event()

        def cb(msg):
            with self._lock:
                self._latest[key] = msg
            self._events[key].set()
            if callback:
                callback(msg)

        self.node.create_subscription(msg_type, topic, cb, qos)

    def _get(self, key, timeout):
        """Return the latest message for key, waiting up to timeout seconds."""
        if timeout and not self._events[key].is_set():
            self._events[key].wait(timeout)
        with self._lock:
            return self._latest.get(key)

    def _wait_fresh(self, key, timeout):
        """Wait for a message that arrives after this call."""
        self._events[key].clear()
        self._events[key].wait(timeout)
        with self._lock:
            return self._latest.get(key)

    # ------------------------------------------------------------------ #
    # Movement
    # ------------------------------------------------------------------ #
    def set_velocity(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """Send a single velocity command (non-blocking).

        linear_x: forward(+)/backward(-) m/s
        linear_y: left(+)/right(-) m/s
        angular_z: anticlockwise(+)/clockwise(-) rad/s

        The robot keeps moving until stop() or another command is sent.
        """
        msg = Twist()
        msg.linear.x = float(_clamp(linear_x, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        msg.linear.y = float(_clamp(linear_y, -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED))
        msg.angular.z = float(_clamp(angular_z, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED))
        self._cmd_vel_pub.publish(msg)

    def move(self, duration, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """Drive with the given velocities for duration seconds, then stop."""
        end = time.monotonic() + duration
        try:
            while time.monotonic() < end:
                self.set_velocity(linear_x, linear_y, angular_z)
                time.sleep(min(1.0 / CMD_RATE_HZ, max(0.0, end - time.monotonic())))
        finally:
            self.stop()

    def stop(self):
        """Stop all wheel motion."""
        self._cmd_vel_pub.publish(Twist())

    def move_forward(self, duration, direction='forward', speed=None):
        """Drive straight for duration seconds.

        direction: 'forward' or 'backward'
        speed: m/s (default self.default_speed)
        """
        speed = abs(speed if speed is not None else self.default_speed)
        if direction not in ('forward', 'backward'):
            raise ValueError("direction must be 'forward' or 'backward'")
        self.move(duration, linear_x=speed if direction == 'forward' else -speed)

    def move_backward(self, duration, speed=None):
        """Drive backwards for duration seconds."""
        self.move_forward(duration, 'backward', speed)

    def move_sideways(self, duration, direction='left', speed=None):
        """Strafe sideways (mecanum) for duration seconds.

        direction: 'left' or 'right'
        """
        speed = abs(speed if speed is not None else self.default_speed)
        if direction not in ('left', 'right'):
            raise ValueError("direction must be 'left' or 'right'")
        self.move(duration, linear_y=speed if direction == 'left' else -speed)

    def move_diagonal(self, duration, forward=True, left=True, speed=None):
        """Move diagonally (mecanum) for duration seconds."""
        speed = abs(speed if speed is not None else self.default_speed) / math.sqrt(2)
        self.move(duration,
                  linear_x=speed if forward else -speed,
                  linear_y=speed if left else -speed)

    def rotate(self, duration, direction='clockwise', speed=None):
        """Spin on the spot for duration seconds.

        direction: 'clockwise' or 'anticlockwise'
        speed: rad/s (default self.default_turn_speed)
        """
        speed = abs(speed if speed is not None else self.default_turn_speed)
        if direction not in ('clockwise', 'anticlockwise'):
            raise ValueError("direction must be 'clockwise' or 'anticlockwise'")
        self.move(duration, angular_z=-speed if direction == 'clockwise' else speed)

    def rotate_degrees(self, degrees, speed=None, timeout=10.0):
        """Rotate by an angle using the odometry yaw for feedback.

        Positive degrees = anticlockwise, negative = clockwise.
        """
        speed = abs(speed if speed is not None else self.default_turn_speed)
        start = self.get_yaw()
        if start is None:
            raise RuntimeError('No odometry received; is the bringup running?')
        target = math.radians(degrees)
        turned, last = 0.0, start
        end = time.monotonic() + timeout
        try:
            while abs(turned) < abs(target) and time.monotonic() < end:
                remaining = abs(target) - abs(turned)
                # slow down over the last ~20 degrees to limit overshoot
                w = max(0.3, min(speed, remaining * 2.0))
                self.set_velocity(angular_z=math.copysign(w, target))
                time.sleep(1.0 / CMD_RATE_HZ)
                yaw = self.get_yaw()
                turned += math.atan2(math.sin(yaw - last), math.cos(yaw - last))
                last = yaw
        finally:
            self.stop()

    def set_motor_speeds(self, m1, m2, m3, m4):
        """Set the four wheel motors directly in revolutions per second.

        Bypasses the kinematics; note the right-side motors are mounted
        mirrored (see controller/mecanum.py). Call stop_motors() to stop.
        """
        msg = MotorsState()
        msg.data = [MotorState(id=i + 1, rps=float(v)) for i, v in enumerate((m1, m2, m3, m4))]
        self._motor_pub.publish(msg)

    def stop_motors(self):
        """Set all four motors to zero rps."""
        self.set_motor_speeds(0, 0, 0, 0)

    # ------------------------------------------------------------------ #
    # IMU / odometry
    # ------------------------------------------------------------------ #
    def get_imu_sensor(self, timeout=1.0):
        """Return the filtered IMU reading as a dict, or None.

        {
          'accel': (x, y, z) m/s^2,
          'gyro': (x, y, z) rad/s,
          'orientation': (x, y, z, w) quaternion,
          'rpy': (roll, pitch, yaw) degrees,
        }
        """
        msg = self._get('imu', timeout)
        if msg is None:
            return None
        a, g, q = msg.linear_acceleration, msg.angular_velocity, msg.orientation
        roll, pitch, yaw = _quaternion_to_rpy(q)
        return {
            'accel': (a.x, a.y, a.z),
            'gyro': (g.x, g.y, g.z),
            'orientation': (q.x, q.y, q.z, q.w),
            'rpy': (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
        }

    def get_imu_raw(self, timeout=1.0):
        """Return the uncalibrated IMU reading: {'accel': (...), 'gyro': (...)}."""
        msg = self._get('imu_raw', timeout)
        if msg is None:
            return None
        a, g = msg.linear_acceleration, msg.angular_velocity
        return {'accel': (a.x, a.y, a.z), 'gyro': (g.x, g.y, g.z)}

    def get_orientation(self, timeout=1.0):
        """Return (roll, pitch, yaw) in degrees from the IMU filter."""
        msg = self._get('imu_rpy', timeout)
        if msg is None:
            return None
        v = msg.vector
        return math.degrees(v.x), math.degrees(v.y), math.degrees(v.z)

    def get_odometry(self, timeout=1.0):
        """Return fused odometry as a dict, or None.

        {'x': m, 'y': m, 'yaw': degrees, 'vx': m/s, 'vy': m/s, 'wz': rad/s}
        """
        msg = self._get('odom', timeout)
        if msg is None:
            return None
        p, t = msg.pose.pose, msg.twist.twist
        _, _, yaw = _quaternion_to_rpy(p.orientation)
        return {
            'x': p.position.x, 'y': p.position.y, 'yaw': math.degrees(yaw),
            'vx': t.linear.x, 'vy': t.linear.y, 'wz': t.angular.z,
        }

    def get_yaw(self, timeout=1.0):
        """Return the odometry heading in radians, or None."""
        msg = self._get('odom', timeout)
        if msg is None:
            return None
        return _quaternion_to_rpy(msg.pose.pose.orientation)[2]

    # ------------------------------------------------------------------ #
    # Lidar
    # ------------------------------------------------------------------ #
    def get_lidar_scan(self, timeout=1.0):
        """Return the latest LaserScan as a dict, or None.

        {'angles': [rad...], 'ranges': [m...], 'range_min': m, 'range_max': m}
        Angle 0 is straight ahead, increasing anticlockwise (to the left).
        Invalid readings are reported as float('inf').
        """
        msg = self._get('scan', timeout)
        if msg is None:
            return None
        ranges = [r if msg.range_min <= r <= msg.range_max else float('inf')
                  for r in msg.ranges]
        angles = [msg.angle_min + i * msg.angle_increment for i in range(len(ranges))]
        return {'angles': angles, 'ranges': ranges,
                'range_min': msg.range_min, 'range_max': msg.range_max}

    def get_distance(self, angle=0.0, width=10.0, timeout=1.0):
        """Return the closest lidar distance (m) in a cone.

        angle: centre of the cone in degrees (0 = ahead, 90 = left,
               -90 = right, 180 = behind)
        width: cone width in degrees
        Returns float('inf') if nothing is detected, None if no scan.
        """
        scan = self.get_lidar_scan(timeout)
        if scan is None:
            return None
        centre, half = math.radians(angle), math.radians(width) / 2
        best = float('inf')
        for a, r in zip(scan['angles'], scan['ranges']):
            diff = math.atan2(math.sin(a - centre), math.cos(a - centre))
            if abs(diff) <= half and r < best:
                best = r
        return best

    def get_distance_ahead(self, width=20.0):
        return self.get_distance(0.0, width)

    def get_distance_behind(self, width=20.0):
        return self.get_distance(180.0, width)

    def get_distance_left(self, width=20.0):
        return self.get_distance(90.0, width)

    def get_distance_right(self, width=20.0):
        return self.get_distance(-90.0, width)

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #
    def _ensure_camera(self, key, topic):
        if key in self._camera_subscribed:
            return
        from cv_bridge import CvBridge
        self._bridge = self._bridge or CvBridge()
        self._subscribe(key, Image, topic)
        self._camera_subscribed.add(key)

    def get_camera_image(self, timeout=5.0):
        """Return the latest RGB camera frame as a BGR numpy array (OpenCV)."""
        self._ensure_camera('rgb_image', RGB_IMAGE_TOPIC)
        msg = self._get('rgb_image', timeout)
        if msg is None:
            return None
        return self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def get_depth_image(self, timeout=5.0):
        """Return the latest depth frame as a numpy array (uint16, millimetres)."""
        self._ensure_camera('depth_image', DEPTH_IMAGE_TOPIC)
        msg = self._get('depth_image', timeout)
        if msg is None:
            return None
        return self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def take_photo(self, path='photo.jpg'):
        """Save the current camera frame to path. Returns True on success."""
        return self.save_camera_image(path) is not None

    def save_camera_image(self, path=None, timeout=5.0):
        """Save the current RGB frame. path defaults to a timestamped .jpg in
        static/photos. Returns the saved path, or None if no frame arrived."""
        import cv2
        frame = self.get_camera_image(timeout)
        if frame is None:
            return None
        path = path or _photo_path('camera', '.jpg')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, frame) else None

    def save_depth_image(self, path=None, colorize=False, timeout=5.0):
        """Save the current depth frame. path defaults to a timestamped .png in
        static/photos. By default the raw 16-bit depth (millimetres) is saved
        losslessly; colorize=True saves a colour-mapped 8-bit image for viewing
        instead (near = red, far = blue, no reading = black). Returns the saved
        path, or None if no frame arrived."""
        import cv2
        import numpy as np
        depth = self.get_depth_image(timeout)
        if depth is None:
            return None
        if colorize:
            valid = depth > 0
            image = np.zeros((*depth.shape, 3), np.uint8)
            if valid.any():
                near, far = depth[valid].min(), depth[valid].max()
                scaled = (depth.astype(np.float32) - near) / max(far - near, 1)
                gray = (255 - np.clip(scaled, 0, 1) * 255).astype(np.uint8)
                image[valid] = cv2.applyColorMap(gray, cv2.COLORMAP_JET)[valid]
        else:
            image = depth
        path = path or _photo_path('depth_color' if colorize else 'depth', '.png')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, image) else None

    # ------------------------------------------------------------------ #
    # Servos / camera gimbal
    # ------------------------------------------------------------------ #
    def set_servo(self, servo_id, pulse, duration=0.5):
        """Move a PWM servo to pulse (500-2500 us, 1500 = centre)."""
        state = PWMServoState()
        state.id = [int(servo_id)]
        state.position = [int(_clamp(pulse, SERVO_MIN, SERVO_MAX))]
        msg = SetPWMServoState()
        msg.state = [state]
        msg.duration = float(duration)
        self._servo_pub.publish(msg)

    def set_servo_angle(self, servo_id, angle, duration=0.5):
        """Move a PWM servo to angle in degrees (-90..90, 0 = centre)."""
        angle = _clamp(angle, -90, 90)
        self.set_servo(servo_id, SERVO_CENTER + angle / 90.0 * 1000, duration)

    def set_camera_pan(self, angle, duration=0.5):
        """Pan the camera gimbal left/right (degrees, 0 = centre)."""
        self.set_servo_angle(CAMERA_PAN_SERVO_ID, angle, duration)

    def set_camera_tilt(self, angle, duration=0.5):
        """Tilt the camera gimbal up/down (degrees, 0 = centre)."""
        self.set_servo_angle(CAMERA_TILT_SERVO_ID, angle, duration)

    def center_camera(self, duration=0.5):
        """Return the camera gimbal to its centre position."""
        self.set_servo(CAMERA_TILT_SERVO_ID, SERVO_CENTER, duration)
        self.set_servo(CAMERA_PAN_SERVO_ID, SERVO_CENTER, duration)

    # ------------------------------------------------------------------ #
    # LEDs, buzzer, OLED
    # ------------------------------------------------------------------ #
    def set_rgb_led(self, index, red, green, blue):
        """Set one RGB LED on the controller board (index 1 or 2)."""
        msg = RGBStates()
        msg.states = [RGBState(index=int(index), red=int(red), green=int(green), blue=int(blue))]
        self._rgb_pub.publish(msg)

    def turn_on_led(self, red=255, green=255, blue=255, index=None):
        """Turn on the RGB LEDs. index=None sets both LEDs."""
        indices = [1, 2] if index is None else [index]
        msg = RGBStates()
        msg.states = [RGBState(index=i, red=int(red), green=int(green), blue=int(blue))
                      for i in indices]
        self._rgb_pub.publish(msg)

    def turn_off_led(self, index=None):
        """Turn off the RGB LEDs. index=None turns off both."""
        self.turn_on_led(0, 0, 0, index)

    def blink_status_led(self, on_time=0.2, off_time=0.2, repeat=3, led_id=1):
        """Blink the single-colour status LED on the controller board."""
        msg = LedState()
        msg.id = int(led_id)
        msg.on_time = float(on_time)
        msg.off_time = float(off_time)
        msg.repeat = int(repeat)
        self._led_pub.publish(msg)

    def beep(self, frequency=1900, on_time=0.1, off_time=0.1, repeat=1):
        """Sound the buzzer (non-blocking)."""
        msg = BuzzerState()
        msg.freq = int(frequency)
        msg.on_time = float(on_time)
        msg.off_time = float(off_time)
        msg.repeat = int(repeat)
        self._buzzer_pub.publish(msg)

    def set_oled_text(self, line, text):
        """Write text to a line of the OLED display (if fitted)."""
        msg = OLEDState()
        msg.index = int(line)
        msg.text = str(text)
        self._oled_pub.publish(msg)

    # ------------------------------------------------------------------ #
    # Battery / button
    # ------------------------------------------------------------------ #
    def get_battery_voltage(self, timeout=3.0):
        """Return the battery voltage in volts, or None."""
        msg = self._get('battery', timeout)
        return None if msg is None else msg.data / 1000.0

    def get_button(self, timeout=0.0):
        """Return the last button event as (button_id, state), or None.

        state: 0 released, 1 pressed, 2 long press, 3 long press repeat,
               4 released after long press, 5 click, 6 double click,
               7 triple click
        """
        msg = self._get('button', timeout)
        return None if msg is None else (msg.id, msg.state)

    def wait_for_button(self, button_id=None, timeout=None):
        """Block until a button is clicked. Returns (button_id, state) or None."""
        end = None if timeout is None else time.monotonic() + timeout
        while end is None or time.monotonic() < end:
            remaining = None if end is None else max(0.0, end - time.monotonic())
            msg = self._wait_fresh('button', remaining if remaining is not None else 3600)
            if msg is not None and msg.state in (0, 5) and \
                    (button_id is None or msg.id == button_id):
                return msg.id, msg.state
        return None

    def on_button(self, callback):
        """Register callback(button_id, state) called on every button event."""
        self._button_callbacks.append(callback)

    def _on_button(self, msg):
        for cb in self._button_callbacks:
            try:
                cb(msg.id, msg.state)
            except Exception as e:  # don't let user code kill the executor
                self.node.get_logger().error(f'button callback failed: {e}')


if __name__ == '__main__':
    # Quick self-test: read sensors, flash LEDs, beep. Does not move.
    with MentorPi() as robot:
        print('battery  :', robot.get_battery_voltage(), 'V')
        print('imu      :', robot.get_imu_sensor())
        print('rpy      :', robot.get_orientation())
        print('odometry :', robot.get_odometry())
        print('ahead    :', robot.get_distance_ahead(), 'm')
        img = robot.get_camera_image()
        print('camera   :', None if img is None else img.shape)
        depth = robot.get_depth_image()
        print('depth    :', None if depth is None else (depth.shape, depth.dtype))
        print('saved    :', robot.save_camera_image())
        print('saved    :', robot.save_depth_image())
        print('saved    :', robot.save_depth_image(colorize=True))
        robot.turn_on_led(0, 0, 255)
        robot.beep()
        time.sleep(1)
        robot.turn_off_led()

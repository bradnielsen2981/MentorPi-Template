# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Lidar interface for the Hiwonder MentorPi MS200 lidar.
# Scans come straight from the ROS 2 topic published by the robot's bringup. The newest scan is kept
# ready as numpy arrays for distance checks (nearest obstacle, sectors, clearest direction) and for
# drawing a top-down picture of the surroundings.
# Mapping (SLAM) lives in mappinginterface.py, which uses this interface together with the camera.
# Angles are in degrees: 0 = straight ahead, 90 = left, -90 = right, 180 = behind. Distances are in metres.
# Run inside the MentorPi container, where ROS 2 is available.

import sys, os, time, threading, math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

SCAN_TOPIC = '/scan_raw'

# Colours (BGR) for the pictures
SCAN_POINT_COLOUR = (0, 0, 255)
ROBOT_COLOUR = (0, 200, 0)
GRID_COLOUR = (60, 60, 60)


def wrap_angle(angle):
    """Wraps an angle in degrees into the range -180 to 180."""
    return (angle + 180.0) % 360.0 - 180.0

def angle_difference(a, b):
    """Smallest difference a - b between two angles in degrees (-180 to 180). Works on numpy arrays too."""
    return (np.asarray(a) - b + 180.0) % 360.0 - 180.0

def yaw_from_quaternion(q):
    """Heading in degrees (anticlockwise from the map's x axis) from a geometry_msgs Quaternion."""
    return math.degrees(math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

#-------------------------------------------
# The Lidar Object
class LidarInterface():

    def __init__(self, timelimit=20, node=None):
        """
        Initializes the Lidar Interface and subscribes to the lidar scan topic.

        Args:
            timelimit (int, optional): General timelimit setting. Defaults to 20.
            node (rclpy.node.Node, optional): An existing ROS node to share (e.g. MentorPi().node, which is
                                              already being spun). If None, the lidar creates and spins its own.
        """
        self.timelimit = timelimit
        np.set_printoptions(suppress=True) # Disable scientific notation for clarity

        self.status = "Init"
        self.angles = None  # Newest scan: angle of each reading in degrees (-180 to 180)
        self.ranges = None  # Newest scan: distance of each reading in metres (inf = nothing detected)
        self.scan_id = 0
        self.scan_time = 0.0
        self.scanrate = 0.0
        self.max_range = 12.0
        self.scan_expire_time = 1.0 # Seconds before a scan counts as too old to use

        # Thread Locks to prevent data corruption between the main program and the ROS callbacks
        self.scan_lock = threading.Lock()
        self._new_scan = threading.Event()
        self._scan_count = 0
        self._rate_start = time.time()

        # Connect to the lidar topic
        self._subscriptions = []
        self._executor = None
        self._owns_node = node is None
        self._owns_rclpy = False
        try:
            if self._owns_node:
                if not rclpy.ok():
                    rclpy.init()
                    self._owns_rclpy = True
                self.node = Node('lidar_interface')
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self.node)
                self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
                self._spin_thread.start()
            else:
                self.node = node
            self._subscriptions.append(self.node.create_subscription(LaserScan, SCAN_TOPIC, self._scan_callback, qos_profile_sensor_data))
            if not self._new_scan.wait(timeout=5):
                raise TimeoutError("no lidar scans received - is the robot's bringup running?")
            self.status = "Running"
        except Exception as e:
            self.status = "Fail"
            print(f"Lidar could not be accessed: {e}")

    def _scan_callback(self, msg):
        """ROS callback: converts the newest scan into numpy arrays of angles (degrees) and ranges (metres)."""
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges) | (ranges < msg.range_min) | (ranges > msg.range_max)] = np.inf
        angles = wrap_angle(np.degrees(msg.angle_min + np.arange(len(ranges)) * msg.angle_increment))

        with self.scan_lock:
            self.angles, self.ranges = angles, ranges
            self.max_range = msg.range_max
            self.scan_id += 1
            self.scan_time = time.time()

        # Scan rate, averaged over each second
        self._scan_count += 1
        if self.scan_time - self._rate_start >= 1.0:
            self.scanrate = round(self._scan_count / (self.scan_time - self._rate_start), 1)
            self._scan_count, self._rate_start = 0, self.scan_time
        self._new_scan.set()

    #-------------------------------------------
    # Scan data
    def get_scan(self):
        """
        Thread-safe method to get the newest scan.

        Returns:
            tuple: (angles, ranges) numpy arrays - angles in degrees (0 = ahead, 90 = left) and distances in
                   metres (inf = nothing detected). (None, None) if there is no recent scan.
        """
        with self.scan_lock:
            if self.ranges is None or time.time() - self.scan_time > self.scan_expire_time:
                return None, None
            return self.angles.copy(), self.ranges.copy()

    def wait_for_new_scan(self, timeout=1.0):
        """Waits until the next scan arrives (the lidar sends 10 a second). Returns False on timeout."""
        self._new_scan.clear()
        return self._new_scan.wait(timeout=timeout)

    def get_distance(self, angle=0, width=20):
        """
        Finds the nearest obstacle inside a cone.

        Args:
            angle (float, optional): Centre of the cone in degrees (0 = ahead, 90 = left, -90 = right, 180 = behind). Defaults to 0.
            width (float, optional): Total width of the cone in degrees. Defaults to 20.

        Returns:
            float: Distance in metres, inf if nothing is in range, or None if there is no recent scan.
        """
        angles, ranges = self.get_scan()
        if ranges is None:
            return None
        inside = np.abs(angle_difference(angles, angle)) <= width / 2.0
        return float(ranges[inside].min()) if inside.any() else math.inf

    def get_distance_ahead(self, width=20):
        """Nearest obstacle straight ahead in metres (see get_distance)."""
        return self.get_distance(0, width)

    def get_distance_behind(self, width=20):
        """Nearest obstacle behind in metres (see get_distance)."""
        return self.get_distance(180, width)

    def get_distance_left(self, width=20):
        """Nearest obstacle on the left in metres (see get_distance)."""
        return self.get_distance(90, width)

    def get_distance_right(self, width=20):
        """Nearest obstacle on the right in metres (see get_distance)."""
        return self.get_distance(-90, width)

    def get_nearest(self, angle=0, width=360):
        """
        Finds the nearest obstacle and the direction it is in (Hiwonder's lidar following method).

        Args:
            angle (float, optional): Centre of the area to search in degrees. Defaults to 0 (ahead).
            width (float, optional): Width of the area to search in degrees. Defaults to 360 (all around).

        Returns:
            dict: {'found': bool, 'distance': metres, 'angle': degrees}, or None if there is no recent scan.
        """
        angles, ranges = self.get_scan()
        if ranges is None:
            return None
        inside = (np.abs(angle_difference(angles, angle)) <= width / 2.0) & np.isfinite(ranges)
        if not inside.any():
            return {'found': False, 'distance': math.inf, 'angle': None}
        i = np.flatnonzero(inside)[np.argmin(ranges[inside])]
        return {'found': True, 'distance': round(float(ranges[i]), 3), 'angle': round(float(angles[i]), 1)}

    def get_sectors(self, sectors=None):
        """
        Nearest obstacle in each direction around the robot - handy for deciding which way to turn.

        Args:
            sectors (dict, optional): {name: (centre angle, width)} in degrees. Defaults to front, left, right
                                      and back, each 90 degrees wide.

        Returns:
            dict: {name: distance in metres (inf = clear)}, or None if there is no recent scan.
        """
        if sectors is None:
            sectors = {'front': (0, 90), 'left': (90, 90), 'back': (180, 90), 'right': (-90, 90)}
        angles, ranges = self.get_scan()
        if ranges is None:
            return None
        result = {}
        for name, (centre, width) in sectors.items():
            inside = np.abs(angle_difference(angles, centre)) <= width / 2.0
            result[name] = round(float(ranges[inside].min()), 3) if inside.any() else math.inf
        return result

    def is_clear(self, distance=0.5, angle=0, width=40):
        """
        Checks that nothing is closer than a distance inside a cone.

        Args:
            distance (float, optional): Metres that must be free. Defaults to 0.5.
            angle (float, optional): Direction to check in degrees. Defaults to 0 (ahead).
            width (float, optional): Width of the cone in degrees. Defaults to 40.

        Returns:
            bool: True if the cone is clear, False if blocked or there is no recent scan.
        """
        nearest = self.get_distance(angle, width)
        return nearest is not None and nearest > distance

    def get_clearest_direction(self, width=30, min_angle=-180, max_angle=180):
        """
        Finds the direction with the most free space, e.g. for exploring or escaping a corner.

        Args:
            width (float, optional): How wide a gap must be in degrees (about the robot's width at 0.5 m is 30). Defaults to 30.
            min_angle (float, optional): Rightmost direction to consider in degrees. Defaults to -180.
            max_angle (float, optional): Leftmost direction to consider in degrees. Defaults to 180.

        Returns:
            dict: {'angle': degrees, 'distance': metres of free space}, or None if there is no recent scan.
        """
        angles, ranges = self.get_scan()
        if ranges is None:
            return None
        best_angle, best_distance = None, -1.0
        for candidate in range(int(min_angle), int(max_angle) + 1, 5):
            inside = np.abs(angle_difference(angles, candidate)) <= width / 2.0
            if inside.any():
                free = min(float(ranges[inside].min()), self.max_range)
                # Prefer directions closer to straight ahead when two are equally free
                if free > best_distance + 0.01 or (abs(free - best_distance) <= 0.01 and abs(wrap_angle(candidate)) < abs(best_angle)):
                    best_angle, best_distance = wrap_angle(candidate), free
        return None if best_angle is None else {'angle': best_angle, 'distance': round(best_distance, 3)}

    def get_points(self):
        """
        The newest scan as x, y points around the robot.

        Returns:
            numpy.ndarray: Nx2 array of (x, y) in metres - x forwards, y to the left. Empty if there is no recent scan.
        """
        angles, ranges = self.get_scan()
        if ranges is None:
            return np.zeros((0, 2), dtype=np.float32)
        valid = np.isfinite(ranges)
        a = np.radians(angles[valid])
        return np.column_stack((ranges[valid] * np.cos(a), ranges[valid] * np.sin(a)))

    #-------------------------------------------
    # Scan picture
    def get_scan_image(self, size=480, view_range=3.0):
        """
        Draws a top-down picture of the newest scan with the robot in the middle, facing up.

        Args:
            size (int, optional): Width and height of the picture in pixels. Defaults to 480.
            view_range (float, optional): Metres from the robot to the edge of the picture. Defaults to 3.0.

        Returns:
            numpy.ndarray: BGR image.
        """
        image = np.zeros((size, size, 3), dtype=np.uint8)
        centre = size // 2
        scale = centre / view_range # pixels per metre

        # A ring every metre
        for metre in range(1, int(view_range) + 1):
            cv2.circle(image, (centre, centre), int(metre * scale), GRID_COLOUR, 1)
            cv2.putText(image, f"{metre}m", (centre + 3, centre - int(metre * scale) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRID_COLOUR, 1)

        # Forward (x) is up the picture and left (y) is to the left
        for x, y in self.get_points():
            px, py = int(centre - y * scale), int(centre - x * scale)
            if 0 <= px < size and 0 <= py < size:
                cv2.circle(image, (px, py), 2, SCAN_POINT_COLOUR, -1)

        cv2.circle(image, (centre, centre), 6, ROBOT_COLOUR, -1)
        cv2.line(image, (centre, centre), (centre, centre - 14), ROBOT_COLOUR, 2)
        cv2.putText(image, f"{self.scanrate} scans/s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return image

    def get_jpeg_scan_image(self, quality=50, size=480, view_range=3.0):
        """The scan picture (see get_scan_image) encoded as JPEG bytes, ready for a Flask stream."""
        ret, jpeg = cv2.imencode('.jpg', self.get_scan_image(size, view_range), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return jpeg.tobytes() if ret else b''

    def stop(self):
        """
        Releases the lidar subscriptions.
        """
        self.status = "Stop"
        try:
            for subscription in self._subscriptions:
                self.node.destroy_subscription(subscription)
            self._subscriptions.clear()
            if self._owns_node and self.node is not None:
                self._executor.shutdown()
                self._spin_thread.join(timeout=2)
                self.node.destroy_node()
                self.node = None
                if self._owns_rclpy and rclpy.ok():
                    rclpy.shutdown()
        except Exception as e:
            print(f"Lidar shutdown problem: {e}")

# TEST LIDAR CODE
if __name__ == '__main__':
    input("Please press enter to begin standalone lidar test: ")
    LIDAR = LidarInterface()
    show_window = bool(os.environ.get('DISPLAY'))

    try:
        # Scan data: watch the numbers change as you move your hand around the lidar
        # (for mapping, run mappinginterface.py)
        print("\n>>> Scan data for 10 seconds (press q in the window to skip)...")
        endtime = time.time() + 10
        while time.time() < endtime and LIDAR.status == "Running":
            LIDAR.wait_for_new_scan()
            print(f"    sectors: {LIDAR.get_sectors()}  nearest: {LIDAR.get_nearest()}  clearest: {LIDAR.get_clearest_direction()}")
            if show_window:
                cv2.imshow('Lidar Scan', LIDAR.get_scan_image())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            time.sleep(0.4)

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the lidar is released when done
        LIDAR.stop()
        cv2.destroyAllWindows()
        time.sleep(1)
        sys.exit(0)

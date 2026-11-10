# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Mapping interface for the Hiwonder MentorPi.
# Builds a map with Hiwonder's SLAM setup (ros2_ws/src/slam) running in the background while the robot drives
# around. Two methods are available:
#   'slam_toolbox' - 2D lidar SLAM (fast, works in the dark)
#   'rtabmap'      - depth camera + lidar SLAM (better at recognising places it has seen before)
# The map can be read as a picture, followed with get_pose() and saved for Nav2.
# Object tagging uses the camera: things it detects (YOLOv5 objects or colours) are placed on the map using
# their depth distance and the robot's position, so the map remembers where e.g. the chair is.
# Angles are in degrees, distances in metres. Run inside the MentorPi container, where ROS 2 is available.

import sys, os, time, threading, math, subprocess, signal, atexit, json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.time import Time
from rclpy.duration import Duration
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lidarinterface import LidarInterface, yaw_from_quaternion
from camerainterface import CameraInterface

MAP_TOPIC = '/map'
# The SLAM programs publish the map once per update and keep the last one for late subscribers
MAP_QOS = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL, reliability=QoSReliabilityPolicy.RELIABLE)
MAP_FRAME = 'map'
BASE_FRAME = 'base_footprint'
CAMERA_INFO_TOPIC = '/ascamera/camera_publisher/rgb0/camera_info'

# Hiwonder's SLAM launches, without the robot bringup (that is already running)
SLAM_LAUNCH_FILES = {
    'slam_toolbox': '/home/ubuntu/ros2_ws/src/slam/launch/include/slam_base.launch.py',
    'rtabmap': '/home/ubuntu/ros2_ws/src/slam/launch/include/rtabmap.launch.py',
}

# Default folders (<project>/static/maps and <project>/logs)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(PROJECT_DIR, 'static', 'maps')
SLAM_LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'slam.log')

# Camera geometry, used until the camera's own calibration (camera_info) and position (TF) arrive
FRAME_WIDTH = 640
DEFAULT_CAMERA_FOV = 70.0  # Horizontal field of view in degrees
CAMERA_OFFSET = (0.0, 0.0) # Camera position from the robot's centre in metres (x ahead, y left)

# Colours (BGR) for the pictures
ROBOT_COLOUR = (0, 200, 0)
OBJECT_COLOUR = (255, 0, 200)


#-------------------------------------------
# The Mapping Object
class MappingInterface():

    def __init__(self, lidar=None, camera=None, node=None, use_camera=True):
        """
        Initializes the Mapping Interface. Uses the lidar and camera interfaces given, or creates its own.

        Args:
            lidar (LidarInterface, optional): An existing lidar interface to use. If None, one is created.
            camera (CameraInterface, optional): An existing camera interface to use (e.g. ROBOT.CAMERA). If None
                                                and use_camera is True, one is created and started.
            node (rclpy.node.Node, optional): An existing ROS node to share (e.g. MentorPi().node, which is
                                              already being spun). If None, mapping creates and spins its own,
                                              and shares it with the lidar and camera it creates.
            use_camera (bool, optional): Needed for object tagging. Defaults to True.
        """
        self.status = "Init"
        self.method = None
        self.map_msg = None
        self.map_id = 0
        self._slam_process = None
        self._slam_log = None

        # Object tagging
        self.objects = []               # Every object seen: {'label', 'x', 'y', 'sightings', 'last_seen'}
        self.merge_distance = 0.5       # Sightings of the same label closer than this (metres) are the same object
        self.min_sightings = 2          # Sightings needed before an object counts (stops one-off false detections)
        self.max_object_distance = 3.0  # Ignore detections further away than this (depth gets unreliable)
        self.tagging_rate = 0.2         # Seconds between looks at the camera's detections
        self._tagging_thread = None
        self._tagging = False
        self._tagging_tasks = []        # Camera tasks started by tagging, removed again when it stops
        self._camera_fx = None
        self._camera_cx = None
        self._camera_frame = None
        self._camera_offset = None

        # Thread Locks to prevent data corruption between the main program, the tagging thread and the ROS callbacks
        self.map_lock = threading.Lock()
        self.slam_lock = threading.Lock()
        self.objects_lock = threading.Lock()

        # Connect to ROS
        self._subscriptions = []
        self._executor = None
        self._owns_node = node is None
        self._owns_rclpy = False
        self._owns_lidar = lidar is None
        self._owns_camera = camera is None and use_camera
        self.lidar = lidar
        self.camera = camera
        try:
            if self._owns_node:
                if not rclpy.ok():
                    rclpy.init()
                    self._owns_rclpy = True
                self.node = Node('mapping_interface')
                self._executor = MultiThreadedExecutor() # Same setup as MentorPi, so the node can be shared either way
                self._executor.add_node(self.node)
                self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
                self._spin_thread.start()
            else:
                self.node = node

            self._subscriptions.append(self.node.create_subscription(OccupancyGrid, MAP_TOPIC, self._map_callback, MAP_QOS))
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self.node)

            if self._owns_lidar:
                self.lidar = LidarInterface(node=self.node)
            if self._owns_camera:
                self.camera = CameraInterface(node=self.node)
                self.camera.start()
            if self.camera is not None:
                self._subscriptions.append(self.node.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._camera_info_callback, 1))
            self.status = "Ready"
        except Exception as e:
            self.status = "Fail"
            print(f"Mapping could not be set up: {e}")

        atexit.register(self.stop_mapping) # Never leave the SLAM program running after the program ends

    def _map_callback(self, msg):
        """ROS callback: keeps the newest map from the SLAM program."""
        with self.map_lock:
            self.map_msg = msg
            self.map_id += 1

    def _camera_info_callback(self, msg):
        """ROS callback: keeps the camera's focal length and centre, used to turn pixels into directions."""
        scale = FRAME_WIDTH / msg.width if msg.width else 1.0
        if msg.k[0] > 0:
            self._camera_fx, self._camera_cx = msg.k[0] * scale, msg.k[2] * scale
        self._camera_frame = msg.header.frame_id

    #-------------------------------------------
    # Mapping
    def start_mapping(self, method='slam_toolbox', timeout=30):
        """
        Starts building a map. Drive the robot around slowly to fill the map in.
        If another program is already mapping, its map is used instead of starting a second one.

        Args:
            method (str, optional): 'slam_toolbox' (lidar) or 'rtabmap' (depth camera + lidar). Defaults to 'slam_toolbox'.
            timeout (int, optional): Seconds to wait for the first map. Defaults to 30.

        Returns:
            bool: True once the first map has arrived.
        """
        if self.status == "Fail":
            return False
        if method not in SLAM_LAUNCH_FILES:
            print("Method must be one of: " + ", ".join(SLAM_LAUNCH_FILES))
            return False
        with self.slam_lock:
            if self.is_mapping():
                if method != self.method:
                    print(f"Already mapping with {self.method} - stop mapping first to change method")
            else:
                time.sleep(1) # Give ROS a moment to discover a mapping program that is already running
                if self.node.count_publishers(MAP_TOPIC) > 0:
                    print("Mapping is already running in another program - using its map")
                else:
                    print(f"Starting mapping ({method}), output in logs/slam.log")
                    os.makedirs(os.path.dirname(SLAM_LOG_PATH), exist_ok=True)
                    self._slam_log = open(SLAM_LOG_PATH, 'w')
                    env = dict(os.environ, need_compile=os.environ.get('need_compile', 'False'))
                    self._slam_process = subprocess.Popen(['ros2', 'launch', SLAM_LAUNCH_FILES[method]], env=env,
                                                          stdout=self._slam_log, stderr=subprocess.STDOUT)
                    self.method = method

        endtime = time.time() + timeout
        while time.time() < endtime:
            if self.get_map() is not None:
                self.status = "Mapping"
                return True
            if self._slam_process is not None and self._slam_process.poll() is not None:
                print("Mapping stopped unexpectedly - see logs/slam.log")
                return False
            time.sleep(0.2)
        print("No map received yet - is the robot's bringup (odometry) running?")
        return False

    def is_mapping(self):
        """Returns True while this interface's SLAM program is running."""
        return self._slam_process is not None and self._slam_process.poll() is None

    def stop_mapping(self):
        """
        Stops the SLAM program. The last map and the tagged objects are kept, so they can still be read or saved.
        """
        with self.slam_lock:
            process, self._slam_process = self._slam_process, None
            if process is not None and process.poll() is None:
                print("Stopping mapping")
                process.send_signal(signal.SIGINT) # ros2 launch passes Ctrl+C on to the SLAM program
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self._slam_log is not None:
                self._slam_log.close()
                self._slam_log = None
        if self.status == "Mapping":
            self.status = "Ready"

    def get_map(self):
        """
        Thread-safe method to get the newest map.

        Returns:
            dict: {'grid': numpy int8 array (row 0 = bottom of the map; -1 unknown, 0 free, 100 occupied),
                   'resolution': metres per cell, 'origin': (x, y) of the bottom-left corner in metres,
                   'width': cells, 'height': cells}, or None if there is no map yet.
        """
        with self.map_lock:
            msg = self.map_msg
        if msg is None:
            return None
        info = msg.info
        grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)
        return {'grid': grid, 'resolution': info.resolution, 'origin': (info.origin.position.x, info.origin.position.y),
                'width': info.width, 'height': info.height}

    def get_pose(self):
        """
        The robot's position on the map.

        Returns:
            dict: {'x': metres, 'y': metres, 'heading': degrees anticlockwise from the map's x axis}, or None
                  if mapping has not started or the position is not known yet.
        """
        try:
            t = self._tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, Time(), timeout=Duration(seconds=0.1))
        except Exception:
            return None
        p = t.transform.translation
        return {'x': round(p.x, 3), 'y': round(p.y, 3), 'heading': round(yaw_from_quaternion(t.transform.rotation), 1)}

    #-------------------------------------------
    # Object tagging
    def start_object_tagging(self, model=True, colours=None):
        """
        Starts a background thread that places what the camera detects onto the map.
        Mapping must be running (the robot's position is needed). Drive slowly so detections line up.

        Args:
            model (bool, optional): Tag YOLOv5 objects (person, chair, cup...). Defaults to True.
            colours (list, optional): Colour names to tag as well, e.g. ['red', 'blue']. Defaults to None.

        Returns:
            bool: True if tagging started.
        """
        if self.camera is None or self.camera.status != "Running":
            print("Object tagging needs a running camera")
            return False
        self.camera.enable_depth()
        if model and "detect_model" not in self.camera.detection_tasks:
            self.camera.add_detection_task("detect_model")
            self._tagging_tasks.append("detect_model")
        if colours:
            if "detect_colour" not in self.camera.detection_tasks:
                self.camera.add_detection_task("detect_colour")
                self._tagging_tasks.append("detect_colour")
            for colour in colours:
                self.camera.add_detection_colour(colour)

        if not self._tagging:
            self._tagging = True
            self._tagging_thread = threading.Thread(target=self._tagging_loop, daemon=True)
            self._tagging_thread.start()
        return True

    def stop_object_tagging(self):
        """Stops placing detections on the map. Objects found so far are kept."""
        self._tagging = False
        if self._tagging_thread and self._tagging_thread is not threading.current_thread():
            self._tagging_thread.join(timeout=2)
        self._tagging_thread = None
        for task in self._tagging_tasks:
            self.camera.remove_detection_task(task)
        self._tagging_tasks.clear()

    def _tagging_loop(self):
        """
        Background thread: takes each new set of camera detections, works out where they are on the map
        and merges them into self.objects.
        """
        last_model_time = None
        while self._tagging:
            time.sleep(self.tagging_rate)
            pose = self.get_pose()
            if pose is None:
                continue
            data = self.camera.get_detection_data()

            sightings = [] # (label, pixel x, distance in mm)
            model = data.get('detect_model', {})
            if model.get('found') and model.get('time') != last_model_time: # Only use each model result once
                last_model_time = model.get('time')
                for target in model.get('targets', []):
                    x, y, w, h = target['rect']
                    sightings.append((target['class'], x + w / 2, target.get('distance')))
            for colour, found in data.get('detect_colour', {}).items():
                (x, y), _, _ = found['rect']
                sightings.append((colour, x, found.get('distance')))

            for label, px, distance in sightings:
                position = self._sighting_to_map(pose, px, distance)
                if position is not None:
                    self._add_sighting(label, *position)

    def _sighting_to_map(self, pose, px, distance):
        """
        Turns a detection (pixel column and depth distance) into a map position.

        Returns:
            tuple: (x, y) in metres on the map, or None if the distance is missing or too far.
        """
        if distance is None:
            return None
        forward = distance / 1000.0 # The depth camera measures straight-ahead distance, not along the line of sight
        if forward > self.max_object_distance:
            return None
        fx = self._camera_fx or (FRAME_WIDTH / 2) / math.tan(math.radians(DEFAULT_CAMERA_FOV / 2))
        cx = self._camera_cx or FRAME_WIDTH / 2
        left = -(px - cx) * forward / fx # Pixels right of centre are to the robot's right

        offset_x, offset_y = self._get_camera_offset()
        rx, ry = forward + offset_x, left + offset_y # Position relative to the robot's centre
        heading = math.radians(pose['heading'])
        return (pose['x'] + rx * math.cos(heading) - ry * math.sin(heading),
                pose['y'] + rx * math.sin(heading) + ry * math.cos(heading))

    def _get_camera_offset(self):
        """Where the camera sits on the robot (x ahead, y left), from TF if the bringup publishes it."""
        if self._camera_offset is None and self._camera_frame:
            try:
                t = self._tf_buffer.lookup_transform(BASE_FRAME, self._camera_frame, Time(), timeout=Duration(seconds=0.1))
                self._camera_offset = (t.transform.translation.x, t.transform.translation.y)
            except Exception:
                pass
        return self._camera_offset or CAMERA_OFFSET

    def _add_sighting(self, label, x, y):
        """Merges a sighting into the nearest known object with the same label, or adds a new object."""
        with self.objects_lock:
            nearest, nearest_distance = None, self.merge_distance
            for obj in self.objects:
                d = math.hypot(obj['x'] - x, obj['y'] - y)
                if obj['label'] == label and d < nearest_distance:
                    nearest, nearest_distance = obj, d
            if nearest is None:
                self.objects.append({'label': label, 'x': x, 'y': y, 'sightings': 1, 'last_seen': time.time()})
            else:
                # Running average, so the position settles as more sightings come in
                n = nearest['sightings']
                nearest['x'] = (nearest['x'] * n + x) / (n + 1)
                nearest['y'] = (nearest['y'] * n + y) / (n + 1)
                nearest['sightings'] = n + 1
                nearest['last_seen'] = time.time()

    def get_objects(self, label=None):
        """
        Thread-safe method to get the objects placed on the map.

        Args:
            label (str, optional): Only return objects with this label, e.g. 'chair'. Defaults to all.

        Returns:
            list: [{'label', 'x', 'y', 'sightings', 'last_seen'}], only objects seen at least min_sightings times.
        """
        with self.objects_lock:
            return [dict(obj, x=round(obj['x'], 2), y=round(obj['y'], 2)) for obj in self.objects
                    if obj['sightings'] >= self.min_sightings and (label is None or obj['label'] == label)]

    def clear_objects(self):
        """Forgets all tagged objects."""
        with self.objects_lock:
            self.objects.clear()

    #-------------------------------------------
    # Pictures and saving
    def _map_to_image(self, grid):
        """Turns an occupancy grid into a greyscale picture: white free, black occupied, grey unknown (top = north)."""
        image = np.full(grid.shape, 205, dtype=np.uint8)
        image[(grid >= 0) & (grid < 25)] = 254
        image[grid >= 65] = 0
        return np.flipud(image) # Row 0 of the grid is the bottom of the map

    def get_map_image(self, size=480, draw_robot=True, draw_objects=True):
        """
        Draws the map as a picture: white = free, black = walls/obstacles, grey = not explored yet.

        Args:
            size (int, optional): Longest side of the picture in pixels. Defaults to 480.
            draw_robot (bool, optional): Draw the robot's position and heading. Defaults to True.
            draw_objects (bool, optional): Draw the tagged objects with their labels. Defaults to True.

        Returns:
            numpy.ndarray: BGR image (a grey picture if there is no map yet).
        """
        map_data = self.get_map()
        if map_data is None:
            image = np.full((size, size, 3), 205, dtype=np.uint8)
            cv2.putText(image, "No map yet", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            return image

        image = cv2.cvtColor(self._map_to_image(map_data['grid']), cv2.COLOR_GRAY2BGR)
        scale = size / max(map_data['width'], map_data['height'])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        ox, oy = map_data['origin']

        def to_pixel(x, y):
            return (int((x - ox) / map_data['resolution'] * scale),
                    int(image.shape[0] - (y - oy) / map_data['resolution'] * scale))

        if draw_objects:
            for obj in self.get_objects():
                px, py = to_pixel(obj['x'], obj['y'])
                cv2.drawMarker(image, (px, py), OBJECT_COLOUR, cv2.MARKER_DIAMOND, 10, 2)
                cv2.putText(image, obj['label'], (px + 7, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, OBJECT_COLOUR, 1)

        pose = self.get_pose() if draw_robot else None
        if pose is not None:
            px, py = to_pixel(pose['x'], pose['y'])
            heading = math.radians(pose['heading'])
            tip = (int(px + 12 * math.cos(heading)), int(py - 12 * math.sin(heading)))
            cv2.circle(image, (px, py), 5, ROBOT_COLOUR, -1)
            cv2.line(image, (px, py), tip, ROBOT_COLOUR, 2)
        return image

    def get_jpeg_map_image(self, quality=50, size=480):
        """The map picture (see get_map_image) encoded as JPEG bytes, ready for a Flask stream."""
        ret, jpeg = cv2.imencode('.jpg', self.get_map_image(size), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return jpeg.tobytes() if ret else b''

    def save_map(self, name=None, folder=MAP_DIR):
        """
        Saves the newest map in the ROS map format (name.pgm + name.yaml, usable by Nav2) plus name.png to view.
        Tagged objects are saved to name_objects.json.

        Args:
            name (str, optional): File name without extension. Defaults to a timestamp, e.g. map_20261029_101500.
            folder (str, optional): Folder to save into. Defaults to static/maps.

        Returns:
            str: Path of the .yaml file, or None if there is no map yet.
        """
        map_data = self.get_map()
        if map_data is None:
            print("No map to save")
            return None
        name = name or time.strftime('map_%Y%m%d_%H%M%S')
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, name)

        image = self._map_to_image(map_data['grid'])
        cv2.imwrite(base + '.pgm', image)
        cv2.imwrite(base + '.png', self.get_map_image(size=max(image.shape), draw_robot=False))
        ox, oy = map_data['origin']
        with open(base + '.yaml', 'w') as f:
            f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {map_data['resolution']}\n"
                    f"origin: [{ox}, {oy}, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        objects = self.get_objects()
        if objects:
            with open(base + '_objects.json', 'w') as f:
                json.dump(objects, f, indent=2)
        print(f"Map saved to {base}.yaml")
        return base + '.yaml'

    def stop(self):
        """
        Stops tagging and mapping, and releases the lidar, camera and ROS connection if this interface created them.
        """
        self.stop_object_tagging()
        self.stop_mapping()
        self.status = "Stop"
        try:
            if self._owns_camera and self.camera is not None:
                self.camera.stop()
            if self._owns_lidar and self.lidar is not None:
                self.lidar.stop()
            for subscription in self._subscriptions:
                self.node.destroy_subscription(subscription)
            self._subscriptions.clear()
            self._tf_listener = None
            if self._owns_node and self.node is not None:
                self._executor.shutdown()
                self._spin_thread.join(timeout=2)
                self.node.destroy_node()
                self.node = None
                if self._owns_rclpy and rclpy.ok():
                    rclpy.shutdown()
        except Exception as e:
            print(f"Mapping shutdown problem: {e}")

# TEST MAPPING CODE
if __name__ == '__main__':
    input("Please press enter to begin standalone mapping test: ")
    MAPPING = MappingInterface()
    show_window = bool(os.environ.get('DISPLAY'))

    try:
        # Push or drive the robot around slowly to see the map grow and objects appear on it
        print("\n>>> Mapping with object tagging for 60 seconds (press q in the window to finish early)...")
        if MAPPING.start_mapping():
            MAPPING.start_object_tagging()
            endtime = time.time() + 60
            while time.time() < endtime:
                print(f"    pose: {MAPPING.get_pose()}  objects: {[(o['label'], o['x'], o['y']) for o in MAPPING.get_objects()]}")
                if show_window:
                    cv2.imshow('Map', MAPPING.get_map_image())
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(1)
            MAPPING.save_map('mapping_test_map')

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee mapping is stopped and the sensors are released when done
        MAPPING.stop()
        cv2.destroyAllWindows()
        time.sleep(1)
        sys.exit(0)

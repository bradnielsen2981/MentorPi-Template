# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Camera interface for the Hiwonder MentorPi depth camera (ascamera).
# Frames come straight from the ROS 2 topics published by the robot's bringup (no video server needed).
# A background thread runs the requested detection tasks (line, colour, YOLOv5 model, depth) on the
# newest frame, draws the results and keeps the latest detection data ready for the robot code.
# Run inside the MentorPi container, where ROS 2 is available.

import sys, os, time, threading, math
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image

RGB_IMAGE_TOPIC = '/ascamera/camera_publisher/rgb0/image'
DEPTH_IMAGE_TOPIC = '/ascamera/camera_publisher/depth0/image_raw'
# Reliable, keep only the newest frame. The camera publishes RELIABLE and its large depth frames
# never arrive on a best-effort (sensor data) subscription.
CAMERA_QOS = 1

# Hiwonder's colour calibration (edit with their lab_tool). 'Stereo' is the table for the ascamera depth camera.
LAB_CONFIG_PATH = '/home/ubuntu/software/lab_tool/lab_config.yaml'
LAB_CAMERA_TYPE = 'Stereo'

# YOLOv5 models installed with Hiwonder's yolov5_ros2 package:
# 'yolov5s' (80 everyday objects - person, cup, chair...), 'traffic_signs_640s_7_0', 'garbage_classification'
YOLO_MODEL_DIR = '/home/ubuntu/ros2_ws/src/yolov5_ros2/config'

# Default folder for saved pictures (<project>/static/photos)
PHOTO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'photos')

DETECTION_TASKS = ['detect_line', 'detect_colour', 'detect_model', 'detect_depth']
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def image_msg_to_array(msg):
    """
    Converts a ROS sensor_msgs/Image into a numpy array without going through cv_bridge.

    Args:
        msg (sensor_msgs.msg.Image): A colour (bgr8/rgb8) or depth (16UC1) image message.

    Returns:
        numpy.ndarray: BGR uint8 image for colour, uint16 millimetres for depth (0 = no reading).
    """
    if msg.encoding in ('bgr8', 'rgb8'):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img
    if msg.encoding in ('16UC1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")

def photo_path(prefix, ext):
    """Timestamped file path in static/photos, e.g. camera_20261024_203015_123.jpg."""
    stamp = time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}'
    return os.path.join(PHOTO_DIR, f'{prefix}_{stamp}{ext}')

def largest_contour(contours, min_area):
    """
    Finds the largest contour above a minimum area.

    Returns:
        tuple: (contour, area), or (None, 0) if no contour is big enough.
    """
    best, best_area = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area and area > best_area:
            best, best_area = c, area
    return best, best_area

#-------------------------------------------
# The Camera Object
class CameraInterface():

    def __init__(self, timelimit=20, node=None, use_depth=True):
        """
        Initializes the Camera Interface, loads colour calibration data and subscribes to the camera topics.

        Args:
            timelimit (int, optional): General timelimit setting. Defaults to 20.
            node (rclpy.node.Node, optional): An existing ROS node to share (e.g. MentorPi().node, which is
                                              already being spun). If None, the camera creates and spins its own.
            use_depth (bool, optional): Subscribe to the depth picture so detections include distances. Defaults to True.
        """
        self.timelimit = timelimit
        np.set_printoptions(suppress=True) # Disable scientific notation for clarity

        self.thread = None
        self.model_thread = None
        self.status = "Init"
        self.mode = "rgb" # 'rgb' shows the colour picture, 'depth' shows the colour-mapped depth picture
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8) # Latest processed frame with drawings
        self.frame_id = 0
        self.rgb_frame = None   # Newest raw colour frame (BGR)
        self.depth_frame = None # Newest raw depth frame (uint16 millimetres, 0 = no reading)
        self.rgb_frame_id = 0
        self.framerate = 0.0

        self.detection_data = {}
        self.detection_tasks = []
        self.detection_colours = []
        self.detect_once = False
        self.linecolour = "black"
        self.min_detection_area = 800 # Pixels at the full 640x480 resolution
        self.detection_data_expire_time = 1.0 # Seconds to keep a detection after it is lost (stops flickering)
        self.clear_temp_detection_data = False

        # Line following (Hiwonder method): horizontal strips as (top, bottom, weight), fractions of the frame height.
        # The strip nearest the robot counts most when working out the steering angle.
        self.line_rois = ((0.90, 0.95, 0.7), (0.80, 0.85, 0.2), (0.70, 0.75, 0.1))
        self.line_search_top = 0.4      # Crossing lines are searched for from here down (fraction of height)
        self.crossing_fraction = 0.5    # A row this full of line colour counts as a crossing line

        # Colour detection runs on a smaller copy of the frame (4x fewer pixels)
        self.colour_process_size = (320, 240)
        self.last_colour_centres = {}   # Used to keep following the same blob between frames

        # Depth
        self.use_depth = False
        self.depth_display_range = (200, 4000) # Millimetres mapped from red (near) to blue (far) in depth mode
        self.depth_band = (0.25, 0.70)         # Rows (fractions of height) checked for obstacles, above the floor

        # Neural network detection (YOLOv5 on the Pi's CPU)
        self.detection_model = None
        self.detection_model_name = None
        self.detection_model_labels = []
        self.detection_model_confidence_level = 0.5
        self.detection_model_size = 320 # 320 runs at ~3 frames/second, 640 at ~1 but sees smaller objects
        self.model_framerate = 0.0
        self._model_result = None

        self.output_text = True
        self.output_message = ""
        self.paused = False
        self.drawing = True

        # Thread Locks to prevent data corruption between the main program and the background processing threads
        self.dict_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.frame_ready = threading.Condition(self.frame_lock) # Notified every time a new processed frame is ready
        self.task_lock = threading.Lock()
        self.colours_lock = threading.Lock()
        self.clear_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self._jpeg = None
        self._jpeg_key = None

        # Load the colours for detection
        try:
            with open(LAB_CONFIG_PATH, 'r', encoding='utf-8') as f:
                self.lab_colours = yaml.safe_load(f)['lab'][LAB_CAMERA_TYPE]
        except Exception as e:
            print(f"Could not load lab colours, using empty dictionary. Error: {e}")
            self.lab_colours = {}

        # Connect to the camera topics
        self._rgb_msg = None
        self._depth_msg = None
        self._new_rgb = threading.Event()
        self._subscriptions = []
        self._depth_subscription = None
        self._executor = None
        self._owns_node = node is None
        self._owns_rclpy = False
        try:
            if self._owns_node:
                if not rclpy.ok():
                    rclpy.init()
                    self._owns_rclpy = True
                self.node = Node('camera_interface')
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self.node)
                self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
                self._spin_thread.start()
            else:
                self.node = node
            self._subscriptions.append(self.node.create_subscription(Image, RGB_IMAGE_TOPIC, self._rgb_callback, CAMERA_QOS))
            if use_depth:
                self.enable_depth()
            if not self._new_rgb.wait(timeout=5):
                raise TimeoutError("no camera frames received - is the robot's bringup running?")
            self.status = "Ready"
        except Exception as e:
            self.status = "Fail"
            print(f"Video capture could not be accessed: {e}")

    def _rgb_callback(self, msg):
        """ROS callback: keeps only the newest colour message. Converting it is left to the processing thread."""
        self._rgb_msg = msg
        self._new_rgb.set()

    def _depth_callback(self, msg):
        """ROS callback: keeps only the newest depth message."""
        self._depth_msg = msg

    def enable_depth(self):
        """Subscribes to the depth picture (if not already). Needed for depth mode, detect_depth and distances."""
        if self._depth_subscription is None and self.node is not None:
            self._depth_subscription = self.node.create_subscription(Image, DEPTH_IMAGE_TOPIC, self._depth_callback, CAMERA_QOS)
            self._subscriptions.append(self._depth_subscription)
        self.use_depth = True

    def start(self):
        """
        Starts the background camera threads if the camera initialized successfully.
        """
        if self.status == "Ready":
            self.status = "Running"
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            # The neural network is slow, so it runs in its own thread and never holds up line/colour detection
            self.model_thread = threading.Thread(target=self._model_loop, daemon=True)
            self.model_thread.start()

    def update(self):
        """
        The core loop executed by the background thread. Waits for each new camera frame, runs the active
        detection tasks, draws visual overlays and updates the shared data dictionaries safely.
        """
        local_detection_data = {}
        framecount = 0
        fps_start = time.time()

        while self.status == "Running":
            # Sleep until the camera delivers a new frame (no busy waiting)
            if not self._new_rgb.wait(timeout=0.5):
                continue
            self._new_rgb.clear()
            if self.paused:
                continue

            try:
                rgb = image_msg_to_array(self._rgb_msg)
                depth_msg = self._depth_msg if self.use_depth else None
                depth = image_msg_to_array(depth_msg) if depth_msg is not None else None
            except Exception as e:
                print(f"Bad camera frame: {e}")
                continue

            with self.frame_lock:
                self.rgb_frame = rgb
                self.depth_frame = depth
                self.rgb_frame_id += 1

            # Frame rate, averaged over each second
            currenttime = time.time()
            framecount += 1
            if currenttime - fps_start >= 1.0:
                self.framerate = round(framecount / (currenttime - fps_start), 1)
                framecount, fps_start = 0, currenttime

            # Clear the temporary detection data when required
            with self.clear_lock:
                if self.clear_temp_detection_data:
                    local_detection_data.clear()
                    self.last_colour_centres.clear()
                    self.clear_temp_detection_data = False

            with self.task_lock:
                current_tasks = list(self.detection_tasks)

            if "detect_line" in current_tasks:
                _, data = self.detect_line(rgb, colour=self.linecolour)
                local_detection_data['detect_line'] = self._keep_or_expire(local_detection_data.get('detect_line', {}), data, currenttime)

            if "detect_colour" in current_tasks:
                with self.colours_lock:
                    colours = list(self.detection_colours)
                found = local_detection_data.setdefault('detect_colour', {})
                for colour, data in self.detect_colours(rgb, colours).items():
                    data = self._keep_or_expire(found.get(colour, {}), data, currenttime)
                    if data.get('found'):
                        if depth is not None:
                            (x, y), (w, h), _ = data['rect']
                            data['distance'] = self.get_distance_at(x, y, radius=max(3, min(w, h) // 4), depth=depth)
                        found[colour] = data
                    else:
                        found.pop(colour, None)
                for colour in list(found):
                    if colour not in colours:
                        del found[colour]

            if "detect_model" in current_tasks:
                result = self._model_result
                if result is not None:
                    data = self._keep_or_expire(local_detection_data.get('detect_model', {}), result, currenttime)
                    if depth is not None:
                        for target in data.get('targets', []):
                            x, y, w, h = target['rect']
                            target['distance'] = self.get_distance_at(x + w // 2, y + h // 2, radius=max(3, min(w, h) // 4), depth=depth)
                    local_detection_data['detect_model'] = data

            if "detect_depth" in current_tasks and depth is not None:
                local_detection_data['detect_depth'] = self.detect_depth(depth)

            # Remove data for tasks that have been switched off
            for task in list(local_detection_data):
                if task not in current_tasks:
                    del local_detection_data[task]

            # Detect once: stop after one full round of results (the model result arrives later than the others)
            if self.detect_once and current_tasks:
                if "detect_model" not in current_tasks or 'detect_model' in local_detection_data:
                    with self.task_lock:
                        self.detection_tasks.clear()
                    self.detect_once = False

            # Choose the picture to show and draw on
            if self.mode == "depth":
                display = self.colourise_depth(depth) if depth is not None else np.zeros_like(rgb)
            else:
                display = rgb.copy()

            if self.drawing and current_tasks:
                self._draw_detections(display, local_detection_data)

            if self.output_text:
                self._draw_text(display, local_detection_data, current_tasks)

            # Update state variables safely
            with self.dict_lock:
                self.detection_data = {k: (dict(v) if isinstance(v, dict) else v) for k, v in local_detection_data.items()}
            with self.frame_ready:
                self.frame = display
                self.frame_id += 1
                self.frame_ready.notify_all()

    def _keep_or_expire(self, old, new, now):
        """Returns the new data if something was found, otherwise keeps the old find until it expires."""
        if new.get('found'):
            return new
        if old.get('found') and now - old.get('time', 0) < self.detection_data_expire_time:
            return old
        return new

    def detect_line(self, frame, colour=None):
        """
        Hiwonder's line following method. Looks for the line colour in three thin horizontal strips near the
        bottom of the frame and combines their centres (nearest strip counts most) into a steering angle.
        Much cheaper than searching the whole picture, and works on curves. Also looks for a crossing line
        (a line running across the path), useful for stopping at junctions.

        Args:
            frame (numpy.ndarray): The current BGR video frame.
            colour (str, optional): Line colour from the YAML file (e.g. 'black'). If None or unknown,
                                    a dark line on a light floor is assumed.

        Returns:
            tuple: (frame, data_dictionary). When found, data contains:
                   'angle'    - steering angle in degrees (+ = line is to the left, turn left; - = turn right)
                   'offset'   - pixels from the frame centre to the line (- = left, + = right)
                   'points'   - centre of the line in each strip, nearest first
                   'line'     - (nearest point, furthest point)
                   'crossing' - {'found': True, 'y': row} if a line runs across the path
        """
        h, w = frame.shape[:2]
        top = int(min(self.line_search_top, min(r[0] for r in self.line_rois)) * h)
        mask = self._line_mask(frame[top:], colour)

        points = []
        weighted_sum, weight_used = 0.0, 0.0
        for r_top, r_bottom, weight in self.line_rois:
            y0, y1 = int(r_top * h) - top, int(r_bottom * h) - top
            contours, _ = cv2.findContours(mask[y0:y1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour, _ = largest_contour(contours, 30)
            if contour is not None:
                (cx, cy), _, _ = cv2.minAreaRect(contour)
                points.append((int(cx), int(cy) + y0 + top))
                weighted_sum += cx * weight
                weight_used += weight

        data = {'found': False}
        if points:
            centre = weighted_sum / weight_used
            angle = math.degrees(-math.atan((centre - w / 2.0) / (h / 2.0)))
            data = {'found': True, 'angle': round(angle, 1), 'offset': int(centre - w / 2),
                    'points': points, 'line': (points[0], points[-1]), 'time': time.time()}

        # Crossing line: the lowest (nearest) row that is mostly line colour
        coverage = mask.mean(axis=1) / 255.0
        rows = np.where(coverage > self.crossing_fraction)[0]
        if rows.size:
            data.update({'found': True, 'crossing': {'found': True, 'y': int(rows.max()) + top}, 'time': time.time()})
            data.setdefault('points', [])

        return frame, data

    def _line_mask(self, region, colour):
        """Binary mask of the line colour in a region of the frame."""
        if colour and colour in self.lab_colours:
            lab = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2LAB), (3, 3), 3)
            mask = cv2.inRange(lab, tuple(self.lab_colours[colour]['min']), tuple(self.lab_colours[colour]['max']))
        else:
            grey = cv2.GaussianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            _, mask = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.dilate(cv2.erode(mask, KERNEL), KERNEL)

    def detect_colours(self, frame, colours):
        """
        Detects several colours at once. The frame is shrunk and converted to LAB once, then each colour is
        masked, cleaned up (erode/dilate removes speckles) and the best blob is chosen. If a colour was seen
        last frame, the blob nearest to it is kept so tracking doesn't jump between objects.

        Args:
            frame (numpy.ndarray): The current BGR video frame.
            colours (list): Colour names from the YAML file.

        Returns:
            dict: colour -> data. When found, data contains 'area', 'rect' ((x, y), (width, height), angle)
                  in full-frame pixels, and 'time'.
        """
        results = {}
        if not colours:
            return results
        small = cv2.resize(frame, self.colour_process_size)
        lab = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2LAB), (5, 5), 5)
        scale_x = frame.shape[1] / small.shape[1]
        scale_y = frame.shape[0] / small.shape[0]
        min_area = self.min_detection_area / (scale_x * scale_y)

        for colour in colours:
            if colour not in self.lab_colours:
                continue
            mask = cv2.inRange(lab, tuple(self.lab_colours[colour]['min']), tuple(self.lab_colours[colour]['max']))
            mask = cv2.dilate(cv2.erode(mask, KERNEL), KERNEL)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = [(c, cv2.contourArea(c)) for c in contours]
            candidates = [(c, a) for c, a in candidates if a > min_area]
            if not candidates:
                results[colour] = {'found': False}
                self.last_colour_centres.pop(colour, None)
                continue

            contour, area = max(candidates, key=lambda c_a: c_a[1])
            last = self.last_colour_centres.get(colour)
            if last is not None:
                def dist(c_a):
                    (x, y), _ = cv2.minEnclosingCircle(c_a[0])
                    return math.hypot(x - last[0], y - last[1])
                nearest = min(candidates, key=dist)
                if dist(nearest) < 50: # pixels in the small frame
                    contour, area = nearest

            (bx, by), (bw, bh), angle = cv2.minAreaRect(contour)
            self.last_colour_centres[colour] = (bx, by)
            target_rect = ((int(bx * scale_x), int(by * scale_y)), (int(bw * scale_x), int(bh * scale_y)), int(angle))
            results[colour] = {'found': True, 'area': int(area * scale_x * scale_y), 'rect': target_rect, 'time': time.time()}
        return results

    def detect_color(self, frame, minC, maxC):
        """
        Detects one LAB colour range in a frame (kept for older code; detect_colours is faster for several colours).

        Returns:
            tuple: (frame, data_dictionary) containing detection success, area, bounding rect and timestamp.
        """
        self.lab_colours['_custom'] = {'min': list(minC), 'max': list(maxC)}
        data = self.detect_colours(frame, ['_custom'])['_custom']
        del self.lab_colours['_custom']
        return frame, data

    def detect_depth(self, depth):
        """
        Finds the nearest obstacle in front of the robot using the depth picture. Only a band of rows above
        the floor is checked (see self.depth_band), split into left, centre and right zones.

        Args:
            depth (numpy.ndarray): uint16 depth frame in millimetres (0 = no reading).

        Returns:
            dict: 'nearest' (mm), 'point' (x, y) of the nearest obstacle, 'zones' {'left', 'centre', 'right'} in mm
                  (None where there is no reading) and 'time'.
        """
        h, w = depth.shape
        top = int(self.depth_band[0] * h)
        band = cv2.medianBlur(np.ascontiguousarray(depth[top:int(self.depth_band[1] * h):4, ::4]), 5) # every 4th pixel, de-speckled

        zones = {}
        for name, part in zip(('left', 'centre', 'right'), np.array_split(band, 3, axis=1)):
            valid = part[part > 0]
            zones[name] = int(np.percentile(valid, 5)) if valid.size > 20 else None

        data = {'found': False, 'zones': zones, 'time': time.time()}
        valid = band > 0
        if valid.any():
            masked = np.where(valid, band, np.iinfo(np.uint16).max)
            row, col = np.unravel_index(np.argmin(masked), masked.shape)
            data.update({'found': True, 'nearest': int(band[row, col]), 'point': (int(col * 4), int(row * 4 + top))})
        return data

    def get_distance_at(self, x, y, radius=5, depth=None):
        """
        Returns the distance to whatever is at pixel (x, y), using the median of the depth readings around it.
        The depth picture lines up with the colour picture, so detection coordinates can be used directly.

        Args:
            x, y (int): Pixel coordinates in the 640x480 frame.
            radius (int, optional): Half-size of the square of pixels used. Defaults to 5.

        Returns:
            int: Distance in millimetres, or None if there is no depth reading there.
        """
        if depth is None:
            with self.frame_lock:
                depth = self.depth_frame
        if depth is None:
            return None
        h, w = depth.shape
        x, y, radius = int(x), int(y), int(radius)
        window = depth[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
        valid = window[window > 0]
        return int(np.median(valid)) if valid.size else None

    def colourise_depth(self, depth):
        """
        Turns a depth frame into a colour picture for viewing: near = red, far = blue, no reading = black.
        Uses a fixed range (self.depth_display_range) so colours don't flicker between frames.
        """
        near, far = self.depth_display_range
        scaled = (np.clip(depth, near, far).astype(np.float32) - near) * (255.0 / (far - near))
        image = cv2.applyColorMap((255 - scaled).astype(np.uint8), cv2.COLORMAP_JET)
        image[depth == 0] = 0
        return image

    def _model_loop(self):
        """
        Background thread that runs the neural network on the newest frame whenever 'detect_model' is active.
        Results are picked up by the main loop, which also adds distances.
        """
        last_frame_id = -1
        while self.status == "Running":
            with self.task_lock:
                active = "detect_model" in self.detection_tasks
            with self.model_lock:
                model, labels = self.detection_model, self.detection_model_labels
            if self.paused or not active or model is None:
                time.sleep(0.05)
                continue
            with self.frame_lock:
                rgb, frame_id = self.rgb_frame, self.rgb_frame_id
            if rgb is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            starttime = time.time()
            try:
                results = model.predict(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), size=self.detection_model_size)
                predictions = results.pred[0].cpu().numpy()
            except Exception as e:
                print(f"Model detection failed: {e}")
                time.sleep(1)
                continue

            targets = []
            for x1, y1, x2, y2, score, class_id in predictions:
                if score < self.detection_model_confidence_level:
                    continue
                class_id = int(class_id)
                targets.append({
                    'class': labels[class_id] if class_id < len(labels) else "Unknown",
                    'score': round(float(score), 2),
                    'rect': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], # x, y (top-left), width, height
                })
            elapsed = time.time() - starttime
            self.model_framerate = round(1 / elapsed, 1) if elapsed > 0 else 0.0
            self._model_result = {'found': bool(targets), 'targets': targets, 'time': time.time()}

    def load_detection_model(self, model="yolov5s", size=None, confidence=None):
        """
        Loads one of Hiwonder's YOLOv5 models. It runs on the Pi's CPU: about 3 frames/second at size 320.

        Args:
            model (str, optional): 'yolov5s' (80 everyday objects - person, cup, chair...), 'traffic_signs_640s_7_0',
                                   'garbage_classification', or a path to a .pt file. Defaults to 'yolov5s'.
            size (int, optional): Image size the model works at (320 or 640). Defaults to self.detection_model_size.
            confidence (float, optional): Minimum score (0-1) to report a target.

        Returns:
            bool: True if the model is ready.
        """
        if size:
            self.detection_model_size = size
        if confidence is not None:
            self.detection_model_confidence_level = confidence
        if self.detection_model is not None and self.detection_model_name == model:
            return True

        path = model if model.endswith('.pt') else os.path.join(YOLO_MODEL_DIR, model + '.pt')
        try:
            from yolov5 import YOLOv5 # imported here because loading PyTorch takes a few seconds
            yolo = YOLOv5(path, device='cpu')
            names = yolo.model.names
            labels = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            return False

        with self.model_lock:
            self.detection_model = yolo
            self.detection_model_name = model
            self.detection_model_labels = labels
        self._model_result = None
        return True

    def clear_detection_model(self):
        """Removes the neural network model and its labels from memory."""
        with self.model_lock:
            self.detection_model = None
            self.detection_model_name = None
            self.detection_model_labels = []
        self._model_result = None

    def _draw_detections(self, frame, data):
        """Draws lines, boxes and distances for the current detection data onto the frame."""
        line = data.get('detect_line', {})
        if line.get('found'):
            points = line.get('points', [])
            for point in points:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], False, (255, 0, 0), 2)
            if 'angle' in line:
                cv2.putText(frame, f"angle {line['angle']}", (points[0][0] + 10, points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            crossing = line.get('crossing')
            if crossing:
                cv2.line(frame, (0, crossing['y']), (frame.shape[1], crossing['y']), (255, 0, 255), 2)

        for colour, colour_data in data.get('detect_colour', {}).items():
            if colour_data.get('found'):
                box = np.intp(cv2.boxPoints(colour_data['rect']))
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                (x, y), _, _ = colour_data['rect']
                label = colour if colour_data.get('distance') is None else f"{colour} {colour_data['distance']}mm"
                cv2.putText(frame, label, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        model = data.get('detect_model', {})
        for target in model.get('targets', []):
            x, y, w, h = target['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            label = f"{target['class']} {target['score']:.2f}"
            if target.get('distance') is not None:
                label += f" {target['distance']}mm"
            cv2.putText(frame, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        depth = data.get('detect_depth', {})
        if depth.get('found'):
            cv2.circle(frame, depth['point'], 8, (255, 255, 255), 2)
            cv2.putText(frame, f"{depth['nearest']}mm", (depth['point'][0] + 10, depth['point'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_text(self, frame, data, tasks):
        """Writes the frame rate, mode, custom message and a short summary of each detection onto the frame."""
        lines = [f"FPS: {self.framerate}  Mode: {self.mode}  {self.output_message}"]
        self.output_message = ""
        if 'detect_line' in tasks:
            line = data.get('detect_line', {})
            text = f"angle {line['angle']} offset {line['offset']}" if 'angle' in line else "not found"
            if line.get('crossing'):
                text += f"  crossing y={line['crossing']['y']}"
            lines.append(f"Line: {text}")
        if 'detect_colour' in tasks:
            found = [f"{c} {d.get('distance') or ''}" for c, d in data.get('detect_colour', {}).items()]
            lines.append("Colour: " + (", ".join(found) if found else "none"))
        if 'detect_model' in tasks:
            if self.detection_model is None:
                lines.append("Model: not loaded")
            else:
                targets = data.get('detect_model', {}).get('targets', [])
                lines.append(f"Model ({self.model_framerate} fps): " + (", ".join(t['class'] for t in targets) or "none"))
        if 'detect_depth' in tasks:
            zones = data.get('detect_depth', {}).get('zones', {})
            lines.append("Depth mm: " + "  ".join(f"{k} {v}" for k, v in zones.items()))
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 15 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    def get_frame(self):
        """
        Thread-safe method to get the current processed video frame (colour or depth picture, depending on the mode).

        Returns:
            numpy.ndarray: The current image frame (or a blank frame if unavailable).
        """
        with self.frame_lock:
            # Guarantees a frame format is returned, never None
            if self.frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.frame.copy()

    def get_rgb_frame(self):
        """Returns the newest raw colour frame (BGR, no drawings), or None."""
        with self.frame_lock:
            return None if self.rgb_frame is None else self.rgb_frame.copy()

    def get_depth_frame(self):
        """Returns the newest raw depth frame (uint16 millimetres, 0 = no reading), or None."""
        with self.frame_lock:
            return None if self.depth_frame is None else self.depth_frame.copy()

    def get_jpeg_frame(self, quality=50):
        """
        Thread-safe method to retrieve the current frame encoded as JPEG bytes.
        Ideal for passing to a Flask or web server for remote streaming. If this frame has already been
        sent, it waits for the next one, so a streaming loop runs at the camera's rate instead of
        encoding the same picture over and over.

        Args:
            quality (int, optional): JPEG encoding quality (0-100). Defaults to 50.

        Returns:
            bytes: The JPEG encoded image.
        """
        with self.frame_ready:
            if self._jpeg_key == (self.frame_id, quality) and self.status == "Running":
                self.frame_ready.wait(timeout=0.5)
            frame_to_encode, key = self.frame, (self.frame_id, quality)

        if key != self._jpeg_key or self._jpeg is None:
            ret, jpeg = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ret: # Absolute fallback if encoding fails (should be rare)
                _, jpeg = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
            self._jpeg, self._jpeg_key = jpeg.tobytes(), key
        return self._jpeg

    def _save_image(self, image, path):
        """Writes an image, creating its folder if needed. Returns the path, or None on failure."""
        if image is None:
            return None
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, image) else None

    def save_frame_as_image(self, path=None, drawings=True):
        """
        Saves the current video frame as a JPEG.

        Args:
            path (str, optional): Where to save. Defaults to a timestamped file in static/photos.
            drawings (bool, optional): True saves the frame as shown (detections drawn, depth picture in depth mode),
                                       False saves the plain colour picture. Defaults to True.

        Returns:
            str: The saved path, or None if there was no frame.
        """
        image = self.get_frame() if drawings else self.get_rgb_frame()
        return self._save_image(image, path or photo_path('camera', '.jpg'))

    def save_depth_image(self, path=None, colourise=False):
        """
        Saves the current depth frame as a PNG.

        Args:
            path (str, optional): Where to save. Defaults to a timestamped file in static/photos.
            colourise (bool, optional): False saves the raw 16-bit depth (millimetres) losslessly, for measuring later.
                                        True saves a colour picture for viewing (near = red, far = blue). Defaults to False.

        Returns:
            str: The saved path, or None if there was no depth frame.
        """
        depth = self.get_depth_frame()
        if depth is not None and colourise:
            depth = self.colourise_depth(depth)
        return self._save_image(depth, path or photo_path('depth_colour' if colourise else 'depth', '.png'))

    def set_mode(self, mode):
        """
        Chooses which picture is shown and streamed. Detection always uses the colour picture.

        Args:
            mode (str): 'rgb' for the normal colour picture, 'depth' for the colour-mapped depth picture.
        """
        if mode not in ('rgb', 'depth'):
            print("Mode must be 'rgb' or 'depth'")
            return
        if mode == 'depth':
            self.enable_depth()
        self.mode = mode

    def turn_on_depth_mode(self):
        """Shows the depth picture (near = red, far = blue) instead of the colour picture."""
        self.set_mode('depth')

    def turn_off_depth_mode(self):
        """Goes back to showing the colour picture."""
        self.set_mode('rgb')

    def toggle_mode(self):
        """Switches between the colour and depth pictures. Returns the new mode."""
        self.set_mode('rgb' if self.mode == 'depth' else 'depth')
        return self.mode

    def get_detection_data(self):
        """
        Thread-safe method to retrieve the most recent detection data.

        Returns:
            dict: A copy of the detection dictionary.
        """
        with self.dict_lock:
            return self.detection_data.copy()

    def clear_detection_data(self):
        """Clears out the active detection dictionary securely."""
        with self.dict_lock:
            self.detection_data.clear()
        with self.clear_lock:
            self.clear_temp_detection_data = True
        self._model_result = None

    def add_detection_task(self, task, detect_once=False):
        """
        Adds a specific vision task to the active background polling cycle.

        Args:
            task (str): One of 'detect_line', 'detect_colour', 'detect_model' or 'detect_depth'.
            detect_once (bool, optional): If True, removes the tasks after the first round of results. Defaults to False.
        """
        if task not in DETECTION_TASKS:
            print("Task must be one of: " + ", ".join(DETECTION_TASKS))
            return
        if task == 'detect_depth':
            self.enable_depth()
        if task == 'detect_model' and self.detection_model is None:
            self.load_detection_model()

        with self.task_lock:
            if task not in self.detection_tasks:
                self.detect_once = detect_once
                self.detection_tasks.append(task)

    def set_detection_tasks(self, tasks=None):
        """
        Overrides the current list of detection tasks with a new list.

        Args:
            tasks (list, optional): List of strings representing new tasks. Defaults to empty list.
        """
        self.clear_detection_tasks()
        for task in tasks or []:
            self.add_detection_task(task)

    def clear_detection_tasks(self):
        """Removes all currently active detection tasks from the polling cycle."""
        with self.task_lock:
            self.detection_tasks.clear()

    def detect_all(self, exclude_colours=None):
        """
        Helper method to activate all available detection methods (lines, colours, model, depth).

        Args:
            exclude_colours (list, optional): Colours in the YAML file to ignore. Defaults to ['black', 'white'].
        """
        if exclude_colours is None:
            exclude_colours = ['black', 'white']

        self.add_detection_task('detect_colour')
        self.add_detection_task('detect_line')
        self.add_detection_task('detect_model')
        if self.use_depth:
            self.add_detection_task('detect_depth')

        with self.colours_lock:
            self.detection_colours = [c for c in self.lab_colours if c not in exclude_colours]

    def turn_on_output_text(self):
        """Enables on-screen text overlays for framerate and data."""
        self.output_text = True

    def turn_off_output_text(self):
        """Disables on-screen text overlays."""
        self.output_text = False

    def set_output_message(self, message):
        """
        Sets a custom text string to display on the video feed.

        Args:
            message (str): The text message to display.
        """
        self.output_message = message

    def remove_detection_task(self, task):
        """
        Removes a single specific task from the active task list.

        Args:
            task (str): The name of the task to remove.
        """
        with self.task_lock:
            if task in self.detection_tasks:
                self.detection_tasks.remove(task)

    def end_detection(self):
        """Helper method that stops all tasks, wipes active data, and clears target colors simultaneously."""
        self.clear_detection_tasks()
        self.clear_detection_data()
        self.clear_detection_colours()

    def turn_on_drawing(self):
        """Enables drawing bounding boxes, lines, and rects on the frame."""
        self.drawing = True

    def turn_off_drawing(self):
        """Disables drawing visual elements on the frame to save processing time."""
        self.drawing = False

    def add_detection_colour(self, colour):
        """
        Adds a named color (from the YAML file) to the active searching rotation.

        Args:
            colour (str): The name of the color to search for (e.g., 'red', 'green').
        """
        with self.colours_lock:
            if colour not in self.detection_colours:
                self.detection_colours.append(colour)

    def set_detection_colours(self, colourlist=None):
        """
        Overrides the current color searching rotation with a specific list.

        Args:
            colourlist (list, optional): List of color name strings. Defaults to empty list.
        """
        with self.colours_lock:
            self.detection_colours = list(colourlist or [])

    def set_line_detection_colour(self, colour):
        """
        Sets a specific mask color for the line detection tracking.

        Args:
            colour (str): Color string name from the YAML settings.
        """
        self.linecolour = colour

    def remove_detection_colour(self, colour):
        """
        Removes a single specific color from the searching rotation.

        Args:
            colour (str): The string name of the color to stop tracking.
        """
        with self.colours_lock:
            if colour in self.detection_colours:
                self.detection_colours.remove(colour)

    def clear_detection_colours(self):
        """Stops the robot from tracking any specific colors by emptying the list."""
        with self.colours_lock:
            self.detection_colours.clear()

    def create_detection_window(self):
        """Opens a standard OpenCV GUI window named 'Detection Mode' on the host screen."""
        cv2.namedWindow('Detection Mode')
        cv2.resizeWindow('Detection Mode', 640, 480)

    def test_detection_performance(self, duration_per_task=5, show_window=True):
        """
        A standalone profiler to test the camera's performance without needing the full RobotInterface.
        Cycles through a baseline, each detection algorithm and depth mode, printing the frame rate of each.

        Args:
            duration_per_task (int): How long to run each task in seconds. Defaults to 5.
            show_window (bool): Whether to render the OpenCV window. Defaults to True.
        """
        tasks_to_test = ['baseline', 'detect_line', 'detect_colour', 'detect_model', 'detect_depth', 'depth_mode']

        # Pre-load dependencies
        self.set_detection_colours(['red', 'green', 'blue'])
        self.load_detection_model()
        self.turn_on_output_text()

        if show_window:
            self.create_detection_window()

        print("\n========================================")
        print(" STARTING STANDALONE CAMERA PROFILER")
        print("========================================\n")

        abort_test = False

        for task in tasks_to_test:
            if abort_test:
                break

            # Clear previous tasks and data
            self.clear_detection_tasks()
            self.clear_detection_data()
            self.set_mode('rgb')

            if task == 'baseline':
                print(f">>> Testing 'Baseline (No Detection)' for {duration_per_task} seconds...")
                message = "BASELINE - RAW FEED"
            elif task == 'depth_mode':
                print(f">>> Testing 'Depth mode' for {duration_per_task} seconds...")
                self.set_mode('depth')
                self.add_detection_task('detect_depth')
                message = "DEPTH MODE"
            else:
                print(f">>> Testing '{task}' for {duration_per_task} seconds...")
                self.add_detection_task(task)
                message = task.upper()

            endtime = time.time() + duration_per_task

            # Run the specific task for the allotted time
            while (time.time() < endtime) and self.status == "Running":
                self.set_output_message(message)
                if show_window:
                    frame = self.get_frame()
                    if frame.size > 0:
                        cv2.imshow('Detection Mode', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        abort_test = True
                        break
                else:
                    # CPU saver for headless mode
                    time.sleep(0.03)

            extra = f", model {self.model_framerate} fps" if task == 'detect_model' else ""
            print(f"    frame rate: {self.framerate} fps{extra}")
            print(f"    data: {self.get_detection_data()}")

            if abort_test:
                print("\n[!] Test aborted by user.")
                break

        print("\n========================================")
        print(" PROFILER COMPLETE ")
        print("========================================\n")

        # Clean up
        self.set_output_message("")
        self.set_mode('rgb')
        self.end_detection()

        if show_window:
            cv2.destroyAllWindows()

        return

    def stop(self):
        """
        Initiates a graceful shutdown of the background camera threads and releases the camera subscriptions.
        """
        self.status = "Stop"
        with self.frame_ready:
            self.frame_ready.notify_all()
        for thread in (self.thread, self.model_thread):
            if thread and thread is not threading.current_thread():
                thread.join(timeout=2)
        try:
            for subscription in self._subscriptions:
                self.node.destroy_subscription(subscription)
            self._subscriptions.clear()
            self._depth_subscription = None
            if self._owns_node and self.node is not None:
                self._executor.shutdown()
                self._spin_thread.join(timeout=2)
                self.node.destroy_node()
                self.node = None
                if self._owns_rclpy and rclpy.ok():
                    rclpy.shutdown()
        except Exception as e:
            print(f"Camera shutdown problem: {e}")

    def pause(self):
        """Temporarily halts the processing of new frames in the background thread."""
        print("Pausing Camera")
        self.paused = True

    def resume(self):
        """Resumes the processing of frames in the background thread."""
        print("Resuming Camera")
        self.paused = False

# TEST CAMERA CODE
if __name__ == '__main__':
    input("Please press enter to begin standalone camera test: ")
    CAMERA = CameraInterface()
    print("\033c")

    try:
        # 1. Start the background threads
        CAMERA.start()
        time.sleep(2) # Give the camera and threads a moment to initialize

        # 2. Let the built-in profiler take over! (show_window needs a display, e.g. VNC)
        CAMERA.test_detection_performance(duration_per_task=10, show_window=bool(os.environ.get('DISPLAY')))

    except KeyboardInterrupt:
        # Catch Ctrl+C gracefully
        print("\n[!] Interrupted by user.")
    finally:
        # Guarantee the camera releases the video feed when done
        CAMERA.end_detection()
        CAMERA.stop()
        cv2.destroyAllWindows()
        time.sleep(1)
        sys.exit(0)

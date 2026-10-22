# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# As soon as the Hiwonder robot starts, an mpg server is running on port 8080
# any time we want we can get the frame from the stream, and detect objects and lines

import sys, os, time, queue, threading, math
import cv2
import numpy as np

sys.path.append('/home/pi/MasterPi')
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import yaml_handle # lab colours

# TRY TO LOAD YOLO MODEL DETECTION USING EDGE TPU
try:
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
    from pycoral.adapters import classify
    from pycoral.utils.edgetpu import list_edge_tpus
    tpu_devices = list_edge_tpus()

    if tpu_devices and len(tpu_devices) > 0:
        MODELDETECTION_ENABLED = True
        print("Edge TPU devices found:", tpu_devices)
    else:
        print("No Edge TPU devices detected.")
        MODELDETECTION_ENABLED = False
except ImportError as e:
    print(f"PyCoral not installed or error importing. Model detection disabled. Error: {e}")
    MODELDETECTION_ENABLED = False


def get_max_contour(contours, min_area=400):
    """
    Finds the largest contour in a given list of contours that exceeds a minimum area threshold.
    
    Args:
        contours (list): A list of OpenCV contours.
        min_area (int, optional): Minimum area required to be considered a valid contour. Defaults to 400.
        
    Returns:
        tuple: (area_max_contour, contour_area_max) where area_max_contour is the largest valid contour 
               (or an empty list if none found), and contour_area_max is its area.
    """
    contour_area_max = 0
    area_max_contour = [] # Return empty list instead of None
    
    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max and contour_area_temp > min_area:
            contour_area_max = contour_area_temp
            area_max_contour = c
            
    return area_max_contour, contour_area_max

def format_dict_with_line_breaks(d, indent=0):
    """
    Recursively formats a dictionary into a highly readable string with indentation and line breaks.
    Useful for printing detection data onto the video frame.
    
    Args:
        d (dict): The dictionary to format.
        indent (int, optional): The current indentation level (spaces). Defaults to 0.
        
    Returns:
        str: A formatted string representation of the dictionary.
    """
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{' ' * indent}{key}:")
            lines.append(format_dict_with_line_breaks(value, indent + 4))
        else:
            lines.append(f"{' ' * indent}{key}: {value}")
    return "\n".join(lines)

#-------------------------------------------
# The Camera Object
class CameraInterface():

    def __init__(self, timelimit=20):
        """
        Initializes the Camera Interface, defines detection variables, 
        loads color calibration data from YAML, and attempts to open the video stream.
        
        Args:
            timelimit (int, optional): General timelimit setting. Defaults to 20.
        """
        self.timelimit = timelimit
        np.set_printoptions(suppress=True) # Disable scientific notation for clarity
        
        self.reader_thread = None
        self.thread = None
        self.status = "Init"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8) # Default blank frame instead of None
        self.raw_frame = None # Holds the absolute newest frame from the stream
        self.new_frame_available = False # Flag to prevent processing the same frame twice
        
        self.detection_task_frameskip = { 'detect_line':3, 'detect_model':5, 'detect_colour':2 }
        self.detection_data = {} 
        self.detection_tasks = []
        self.detection_colours = []
        
        self.detection_model = None
        self.detection_model_labels = []
        self.detection_model_confidence_level = 0.7
        self.input_shape = None
        self.input_details = None
        
        self.detect_once = False
        self.colour_shift = 0 
        self.linecolour = "black"
        self.min_detection_area = 800
        self.detection_data_expire_time = 1.0 
        self.clear_temp_detection_data = False
        
        self.output_text = True
        self.output_message = ""
        self.paused = False
        self.drawing = True
        
        # Thread Locks to prevent data corruption between the main program and the background processing thread
        self.dict_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.task_lock = threading.Lock()
        self.colours_lock = threading.Lock()
        self.clear_lock = threading.Lock()

        # Load the colours for detection
        try:
            self.lab_colours = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
        except Exception as e:
            print(f"Could not load lab colours, using empty dictionary. Error: {e}")
            self.lab_colours = {}
        
        try:
            self.capture = cv2.VideoCapture('http://127.0.0.1:8080?action=stream')
            if not self.capture.isOpened():
                raise ValueError("Stream opened but not reading frames.")
            time.sleep(1)
            self.status = "Ready"
        except Exception as e:
            self.status = "Fail"
            self.capture = None
            print(f"Video capture could not be accessed: {e}")

    def start(self):
        """
        Starts the background camera threads if the camera initialized successfully.
        """
        if self.status == "Ready":
            self.status = "Running"
            self.currenttime = time.time()
            
            # 1. Start the ultra-fast reader thread to prevent lag
            self.reader_thread = threading.Thread(target=self._frame_reader, daemon=True)
            self.reader_thread.start()
            
            # 2. Start the heavy processing thread
            self.thread = threading.Thread(target=self.update, args=(), daemon=True)
            self.thread.start()

    def _frame_reader(self):
        """
        Ultra-fast background thread dedicated ONLY to pulling frames off the network stream.
        This prevents the OpenCV buffer from filling up and causing video lag.
        """
        while self.status == "Running":
            if self.paused:
                time.sleep(0.01)
                continue
                
            try:
                ret, img = self.capture.read()
                if ret and img is not None:
                    with self.frame_lock:
                        self.raw_frame = img
                        self.new_frame_available = True # WE HAVE A NEW IMAGE!
            except Exception:
                continue

    def update(self):
        """
        The core loop executed by the background thread. Processes the newest available frame, 
        calculates framerate, staggers detection tasks to prevent CPU bottlenecks, draws visual overlays, 
        and updates the shared data dictionaries safely.
        """
        time.sleep(2) # Camera warmup

        framecount = 1 
        framerate = 0.0
        local_detection_data = {}

        while self.status == "Running":
            if self.paused:
                time.sleep(0.01)
                continue
            
            # Grab the absolute newest frame ONLY if it is genuinely new
            current_frame = None
            with self.frame_lock:
                if self.new_frame_available and self.raw_frame is not None:
                    current_frame = self.raw_frame.copy()
                    self.new_frame_available = False # Reset the flag!
            
            # If no new frame has arrived yet, sleep for a tiny fraction of a millisecond and check again
            if current_frame is None:
                time.sleep(0.005)
                continue
            
            # Clear the temporary detection data when required
            if self.clear_temp_detection_data:
                with self.clear_lock:
                    local_detection_data.clear()
                    self.colour_shift = 0
                    self.clear_temp_detection_data = False
            
            with self.task_lock:
                # Copying lightweight list to avoid locking during entire processing loop
                current_tasks = list(self.detection_tasks)

            # FRAME SKIP - process average framing
            currenttime = time.time()
            if framecount >= 9:
                framecount = 2
                elapsedtime = currenttime - self.currenttime
                self.currenttime = currenttime
                if elapsedtime > 0:
                    framerate = round(7 / elapsedtime, 2)
            framecount += 1
            
            taskcomplete = False 

            if current_tasks:
                # check for line detection
                if "detect_line" in current_tasks and (framecount % self.detection_task_frameskip['detect_line'] == 0):
                    if 'detect_line' not in local_detection_data:
                        local_detection_data['detect_line'] = {}

                    current_frame, data = self.detect_line(current_frame, threshold=100, colour=self.linecolour)
                    if data.get('found', False):
                        local_detection_data['detect_line'] = data
                    else:
                        if local_detection_data['detect_line'].get('found') and (currenttime - local_detection_data['detect_line'].get('time', 0)) > self.detection_data_expire_time:
                            local_detection_data['detect_line'] = {} 
                    taskcomplete = True  

                # check for colour detection
                elif "detect_colour" in current_tasks and (framecount % self.detection_task_frameskip['detect_colour'] == 0):
                    if 'detect_colour' not in local_detection_data:
                        local_detection_data['detect_colour'] = {}
                    
                    with self.colours_lock:
                        col_count = len(self.detection_colours)
                        if col_count > 1:
                            colour = self.detection_colours[self.colour_shift]
                            self.colour_shift = (self.colour_shift + 1) % col_count
                        elif col_count == 1:
                            colour = self.detection_colours[0]
                        else:
                            colour = None
                            self.colour_shift = 0
                        
                    if colour and colour in self.lab_colours:
                        minC = np.array(self.lab_colours[colour]['min'])
                        maxC = np.array(self.lab_colours[colour]['max'])
                        
                        current_frame, data = self.detect_color(current_frame, minC, maxC)
                        if data.get('found', False):
                            local_detection_data['detect_colour'][colour] = data
                        
                        if colour in local_detection_data['detect_colour']:
                            if local_detection_data['detect_colour'][colour].get('found') and (currenttime - local_detection_data['detect_colour'][colour].get('time', 0)) > self.detection_data_expire_time:
                                del local_detection_data['detect_colour'][colour]
                    taskcomplete = True

                # use neural network for detection              
                elif "detect_model" in current_tasks and (framecount % self.detection_task_frameskip['detect_model'] == 0):
                    if 'detect_model' not in local_detection_data:
                        local_detection_data['detect_model'] = {}

                    current_frame, data = self.detect_model(current_frame)
                    if data.get('found', False):
                        local_detection_data['detect_model'] = data
                    else:
                        if local_detection_data['detect_model'].get('found') and (currenttime - local_detection_data['detect_model'].get('time', 0)) > self.detection_data_expire_time:
                            local_detection_data['detect_model'] = {}
                    taskcomplete = True

                if self.detect_once and taskcomplete:
                    with self.task_lock:
                        self.detection_tasks.clear()
                    self.detect_once = False
                    framecount = 1   

            # Draw detection lines and boxes
            if self.drawing and current_tasks:
                if local_detection_data.get('detect_line', {}).get('found'):
                    start_point = local_detection_data['detect_line']['line'][0]
                    end_point = local_detection_data['detect_line']['line'][1]
                    cv2.line(current_frame, start_point, end_point, (255, 0, 0), 2)

                for color, data in local_detection_data.get('detect_colour', {}).items():
                    if data.get('found'):
                        box = cv2.boxPoints(data['rect'])
                        box = np.int0(box) 
                        cv2.drawContours(current_frame, [box], 0, (0, 255, 0), 2)

                if local_detection_data.get('detect_model', {}).get('found'):
                    for target in local_detection_data['detect_model']['targets']:
                        x_min, y_min, width, height = map(int, target['rect'])
                        x_max, y_max = x_min + width, y_min + height
                        class_label = target['class']
                        confidence = target['score']

                        cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        label = f"Class: {class_label} | {confidence:.2f}"
                        cv2.putText(current_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write text
            if self.output_text:
                cv2.putText(current_frame, f"Frame rate: {framerate}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(current_frame, self.output_message, (300, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                self.output_message = "" 
                
                if current_tasks:
                    formatted_text = format_dict_with_line_breaks(local_detection_data)
                    for i, line in enumerate(formatted_text.split('\n')):
                        cv2.putText(current_frame, line, (10, 30 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)

            # Update state variables safely
            with self.frame_lock:
                self.frame = current_frame
                
            with self.dict_lock:
                self.detection_data = local_detection_data.copy()

    def detect_line(self, frame, threshold=150, colour=None, min_length=240):
        """
        Uses OpenCV Canny edge detection and Probabilistic Hough Transform to find the longest straight line.
        
        Args:
            frame (numpy.ndarray): The current BGR video frame.
            threshold (int, optional): Accumulator threshold parameter. Defaults to 150.
            colour (str, optional): Target color to filter by (e.g., 'black', 'white'). Defaults to None.
            min_length (int, optional): Minimum length of the line in pixels. Defaults to 240.
                                        WARNING: A 640x480 frame has a maximum diagonal of 800 pixels.
            
        Returns:
            tuple: (frame, data_dictionary) containing detection success, coordinates, and timestamp.
        """
        data = {'found': False}
        
        if colour and colour in self.lab_colours:
            minC = np.array(self.lab_colours[colour]['min'])
            maxC = np.array(self.lab_colours[colour]['max'])
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame_mask = cv2.inRange(frame_lab, minC, maxC)
        else:
            frame_mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(frame_mask, 50, 150, apertureSize=3)
        
        # Use probabilistic Hough Transform (HoughLinesP) to get actual line segments
        # minLineLength handles the length filtering natively and efficiently
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi/180, 
            threshold=threshold, 
            minLineLength=min_length, 
            maxLineGap=50  # Allows small breaks in the line (e.g., glare or floor seams)
        )

        if lines is not None:
            longest_line_length_sq = 0
            longest_line = []

            for line in lines:
                # HoughLinesP natively returns the start and end coordinates of the actual segment
                x1, y1, x2, y2 = line[0]

                # Avoid square root for distance comparison to save CPU cycles
                line_length_sq = (x2 - x1)**2 + (y2 - y1)**2

                if line_length_sq > longest_line_length_sq:
                    longest_line_length_sq = line_length_sq
                    longest_line = ((x1, y1), (x2, y2))

            if longest_line:
                data = {'found': True, 'line': longest_line, 'time': time.time()}

        return frame, data
    
    def detect_color(self, frame, minC, maxC):
        """
        Detects specific colors in a frame by converting to LAB color space, masking out ranges, 
        and finding the bounding box of the largest resultant contour.
        
        Args:
            frame (numpy.ndarray): The current BGR video frame.
            minC (numpy.ndarray): The minimum LAB array boundary.
            maxC (numpy.ndarray): The maximum LAB array boundary.
            
        Returns:
            tuple: (frame, data_dictionary) containing detection success, area, bounding rect coordinates, and timestamp.
        """
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(frame_lab, minC, maxC)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour, contour_area_max = get_max_contour(contours, self.min_detection_area)
        data = { 'found': False }
        
        if len(max_contour) > 0: 
            rect = cv2.minAreaRect(max_contour)
            (box_x, box_y), (box_w, box_h), angle = rect
            target_rect = ((int(box_x), int(box_y)), (int(box_w), int(box_h)), int(angle))

            data = { 'found': True, 'area': contour_area_max, 'rect': target_rect, 'time': time.time() }

        return frame, data

    def detect_model(self, frame):
        """
        Uses a Google Coral Edge TPU and a TensorFlow Lite model (e.g., YOLO) to identify objects 
        in the current frame. Crops/Resizes the frame to the model's required input size, processes 
        the tensors, and maps bounding boxes back to the original frame scale.
        
        Args:
            frame (numpy.ndarray): The current BGR video frame.
            
        Returns:
            tuple: (frame, data_dictionary) containing detection success, list of identified targets (with boxes, scores, classes), and timestamp.
        """
        data = {'found': False} 

        # Abort if the model isn't loaded or EdgeTPU isn't running
        if not MODELDETECTION_ENABLED or self.detection_model is None or self.input_details is None:
            return frame, data

        # Get original frame dimensions and required model dimensions
        f_height, f_width, _ = frame.shape
        input_shape = self.input_details[0]['shape']
        i_height, i_width = input_shape[1], input_shape[2] 

        # Prepare the image for the neural network
        scaled_frame = cv2.resize(frame, (i_width, i_height))
        input_data = np.expand_dims(scaled_frame, axis=0)
    
        # Run the EdgeTPU inference
        self.detection_model.set_tensor(self.input_details[0]['index'], input_data)
        self.detection_model.invoke()

        # Retrieve the raw output tensor
        output_details = self.detection_model.get_output_details()
        output_data = np.squeeze(self.detection_model.get_tensor(output_details[0]['index']))

        # Get quantization parameters to convert int8 back to floats
        scale, zero_point = output_details[0]['quantization']

        # Extract relevant columns using NumPy slicing (optimized)
        # YOLO format is [center_x, center_y, width, height]
        center_x = output_data[:, 0]
        center_y = output_data[:, 1]
        width = output_data[:, 2]
        height = output_data[:, 3]
        confidence = ((output_data[:, 4] - zero_point) * scale).astype(np.float32)

        class_probabilities = output_data[:, 5:]
        class_ids = np.argmax(class_probabilities, axis=1)

        # Apply confidence mask early to save processing time on junk data
        mask = confidence > self.detection_model_confidence_level
        center_x, center_y = center_x[mask], center_y[mask]
        width, height = width[mask], height[mask]
        confidence, class_ids = confidence[mask], class_ids[mask]

        # THE FIX: Convert YOLO center coordinates to top-left OpenCV coordinates
        x_min_raw = center_x - (width / 2.0)
        y_min_raw = center_y - (height / 2.0)

        # Scale the coordinates back up to match the original 640x480 video frame
        scale_x = f_width / i_width
        scale_y = f_height / i_height

        x = (x_min_raw * scale_x).astype(int)
        y = (y_min_raw * scale_y).astype(int)
        width = (width * scale_x).astype(int)
        height = (height * scale_y).astype(int)

        # Build the preliminary target list
        targets = [
            {
                'rect': [x[i], y[i], width[i], height[i]], 
                'score': float(confidence[i]), 
                'class': self.detection_model_labels[class_ids[i]] if class_ids[i] < len(self.detection_model_labels) else "Unknown"
            } 
            for i in range(len(x))
        ]

        # Apply Non-Maximum Suppression (NMS) to remove overlapping duplicate boxes
        if targets:
            boxes = np.array([t['rect'] for t in targets])
            scores = np.array([t['score'] for t in targets])

            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.3, nms_threshold=0.4)
            
            if len(indices) > 0:
                # Handle different OpenCV versions returning NMS indices differently
                filtered_targets = [targets[i[0] if isinstance(i, (list, np.ndarray)) else i] for i in indices]

                data['found'] = True
                data['time'] = time.time()
                data['targets'] = filtered_targets

        return frame, data

    def load_detection_model(self, model_file="models/yolov5s-int8-224_edgetpu.tflite", classes_file="models/coco.names"):
        """
        Loads the TFLite neural network model into the PyCoral interpreter and loads COCO human-readable labels.
        """
        if not MODELDETECTION_ENABLED:
            print("Edge TPU not enabled/working.")
            return

        try:
            self.detection_model = make_interpreter(model_file)
            self.detection_model.allocate_tensors()
            self.input_details = self.detection_model.get_input_details()
            
            with open(classes_file, "r") as f:
                self.detection_model_labels = [line.strip() for line in f.readlines()] 
        except Exception as e:
            print(f"Failed to load model or labels: {e}")
            self.detection_model = None

    def get_frame(self):
        """
        Thread-safe method to get the current video frame.
        
        Returns:
            numpy.ndarray: The current image frame (or a blank frame if unavailable).
        """
        with self.frame_lock:
            # Guarantees a frame format is returned, never None
            if self.frame is None:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            return self.frame.copy()
    
    def get_jpeg_frame(self, quality=50):
        """
        Thread-safe method to retrieve the current frame encoded as JPEG bytes.
        Ideal for passing to a Flask or web server for remote streaming.
        
        Args:
            quality (int, optional): JPEG encoding quality (0-100). Defaults to 50.
            
        Returns:
            bytes: The JPEG encoded image.
        """
        with self.frame_lock:
            frame_to_encode = self.frame if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
        ret, jpeg = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ret:
            return jpeg.tobytes()
        
        # Absolute fallback if encoding fails (should be rare)
        _, fallback = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))
        return fallback.tobytes()
    
    def save_frame_as_image(self):
        """Saves the current video frame to the local disk as 'frame.jpg'."""
        with self.frame_lock:
            if self.frame is not None:
                cv2.imwrite('frame.jpg', self.frame)
            
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
    
    def add_detection_task(self, task, detect_once=False):
        """
        Adds a specific vision task to the active background polling cycle.
        
        Args:
            task (str): One of 'detect_line', 'detect_colour', or 'detect_model'.
            detect_once (bool, optional): If True, removes the task after the first successful frame. Defaults to False.
        """
        if task not in ['detect_line', 'detect_colour', 'detect_model']:
            print("Task must be one of: detect_line, detect_colour, detect_model")
            return

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
        if tasks is None:
            tasks = []
        for task in tasks:
            self.add_detection_task(task)
    
    def clear_detection_tasks(self):
        """Removes all currently active detection tasks from the polling cycle."""
        with self.task_lock:
            self.detection_tasks.clear()
    
    def detect_all(self, exclude_colours=None):
        """
        Helper method to instantly activate all available detection methods (lines, models, colors).
        
        Args:
            exclude_colours (list, optional): Colors in the YAML file to ignore. Defaults to ['black', 'white'].
        """
        if exclude_colours is None:
            exclude_colours = ['black', 'white']
            
        self.add_detection_task('detect_colour')
        self.add_detection_task('detect_line')
        self.load_detection_model()
        self.add_detection_task('detect_model')
        
        with self.colours_lock:
            self.detection_colours = list(self.lab_colours.keys())
            for colour in exclude_colours:
                if colour in self.detection_colours:
                    self.detection_colours.remove(colour)

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
        if colourlist is None:
            colourlist = []
        with self.colours_lock:
            self.detection_colours = colourlist
            
    def set_line_detection_colour(self, colour):
        """
        Sets a specific mask color for the line detection tracking.
        
        Args:
            colour (str): Color string name from the YAML settings.
        """
        self.linecolour = colour
        
    def clear_detection_model(self):
        """Removes the TensorFlow Lite model and associated labels from memory."""
        self.detection_model = None
        self.detection_model_labels = []
        
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
        Cycles through a baseline and each detection algorithm.
        
        Args:
            duration_per_task (int): How long to run each task in seconds. Defaults to 5.
            show_window (bool): Whether to render the OpenCV window. Defaults to True.
        """
        tasks_to_test = ['baseline', 'detect_line', 'detect_colour', 'detect_model']
        
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
            
            if task == 'baseline':
                print(f">>> Testing 'Baseline (No Detection)' for {duration_per_task} seconds...")
                self.set_output_message("BASELINE - RAW FEED")
            else:
                print(f">>> Testing '{task}' for {duration_per_task} seconds...")
                self.add_detection_task(task)
                self.set_output_message(task.upper())
            
            endtime = time.time() + duration_per_task
            
            # Run the specific task for the allotted time
            while (time.time() < endtime) and self.status == "Running":
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
            
            if abort_test:
                print("\n[!] Test aborted by user.")
                break 
                
        print("\n========================================")
        print(" PROFILER COMPLETE ")
        print("========================================\n")
        
        # Clean up
        self.set_output_message("")
        self.end_detection()
        
        if show_window:
            cv2.destroyAllWindows()
            
        return
        
    def stop(self):
        """
        Initiates a graceful shutdown of the background camera threads and releases 
        the video capture device bindings.
        """
        self.status = "Stop"
        time.sleep(2)
        if self.capture:
            self.capture.release()
            
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
        
        # 2. Let the built-in profiler take over!
        CAMERA.test_detection_performance(duration_per_task=10, show_window=True)
        
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
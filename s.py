import threading
import speech_recognition as sr
import cv2
from ultralytics import YOLO
import pyttsx3
import time
import datetime
import requests
from geopy.geocoders import Nominatim
import queue
import signal
import sys
import os
import logging
import math
from pathlib import Path
import platform
if platform.system() == 'Windows':
    import winsound
else:
    import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("navigation_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NavigationAssistant")

# CRITICAL: Define all global variables at the module level
# Shared variables for thread communication
detection_active = False
stop_threads = False
command_queue = queue.Queue()
speech_queue = queue.Queue()

# Global variables for system configuration
MODEL_PATH = 'yolov8s.pt'  # Upgraded to YOLOv8s for better accuracy
CONFIDENCE_THRESHOLD = 0.6  # Increased for fewer false positives
FEEDBACK_COOLDOWN = 2.0  # Seconds between any feedback to avoid constant speech
REPEAT_COOLDOWN = 10.0  # Seconds before repeating a warning about the same object
COMMAND_TIMEOUT = 5  # Seconds to wait for command recognition
SKIP_FRAMES = 3  # Process only 1 out of every N frames for better performance
PROXIMITY_THRESHOLD = 0.6  # Objects taking up more than this fraction of frame are "close"

# Critical obstacles that should always trigger warnings
CRITICAL_OBSTACLES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'pothole', 'stairs'}

# How much an obstacle needs to change in size to be considered a "new situation"
SIZE_CHANGE_THRESHOLD = 0.15  # 15% change in relative size indicates significant movement

# Reference height in meters for common objects (used for distance estimation)
REFERENCE_HEIGHTS = {
    'person': 1.7,  # Average human height
    'car': 1.5,     # Average car height
    'truck': 3.0,   # Average truck height
    'bus': 3.2,     # Average bus height
    'bicycle': 1.0, # Average bicycle height
    'motorcycle': 1.2, # Average motorcycle height
    'dog': 0.6,     # Average dog height
    'cat': 0.3,     # Average cat height
    'chair': 0.8,   # Average chair height
    'bench': 0.5,   # Average bench height
    'pothole': 0.3, # Average pothole depth
    'stairs': 0.2,  # Average stair height (per step)
    # Default value for other objects
    'default': 1.0
}

# Focal length estimation (will be calculated after camera initialization)
FOCAL_LENGTH = None  

# Global dictionary to track detected objects between frames
detected_objects = {}

# Global variable for user's current location
current_location = None

# Speech engine lock to prevent multiple instances
speech_engine_lock = threading.Lock()

# Path to save screenshots
SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Add global variable for frame regions
frame_regions = None

def get_current_location():
    """Attempt to get the user's current location using IP-based geolocation"""
    try:
        # Get IP-based location (this is approximate)
        response = requests.get('https://ipinfo.io/json', timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'loc' in data:
                coords = data['loc'].split(',')
                lat, lon = float(coords[0]), float(coords[1])
                
                # Get address from coordinates
                geolocator = Nominatim(user_agent="navigation_assistant")
                location = geolocator.reverse((lat, lon), language='en', timeout=3)
                
                if location:
                    return location.address
                else:
                    return f"Coordinates: {lat}, {lon}"
        
        return "Location information unavailable"
    
    except Exception as e:
        logger.error(f"Error getting location: {e}")
        return "Could not determine location"

def signal_handler(sig, frame):
    """Handle program termination signals"""
    global stop_threads
    logger.info("Termination signal received. Shutting down...")
    stop_threads = True
    sys.exit(0)

def speak_text(text):
    """Add text to speech queue for processing"""
    speech_queue.put(text)

def speech_thread():
    """Thread dedicated to text-to-speech processing"""
    global stop_threads
    try:
        engine = pyttsx3.init()
        
        # Optimize speech engine
        engine.setProperty('rate', 180)  # Speed of speech
        
        # Get available voices and set a clearer voice if available
        voices = engine.getProperty('voices')
        if voices:
            for voice in voices:
                if "english" in voice.name.lower() and "female" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        while not stop_threads:
            try:
                if not speech_queue.empty():
                    # Get text from queue
                    text = speech_queue.get(block=False)
                    
                    # Log speech for debugging
                    logger.info(f"Speaking: {text}")
                    
                    # Speak text
                    with speech_engine_lock:
                        engine.say(text)
                        engine.runAndWait()
                    
                    # Mark task as done
                    speech_queue.task_done()
                else:
                    # Sleep to prevent CPU overuse
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in speech processing: {e}")
                time.sleep(0.1)
    except Exception as e:
        logger.error(f"Fatal error in speech thread: {e}")
        speak_text("Error initializing speech system. Please restart the application.")

def voice_command_thread():
    """Thread for continuously listening to voice commands"""
    global detection_active, stop_threads, current_location
    
    recognizer = sr.Recognizer()
    
    # Flag to track if we need to retry getting location
    location_retry_needed = True
    
    # Attempt to get initial location in background
    threading.Thread(target=lambda: globals().update(current_location=get_current_location()), 
                    daemon=True).start()
    
    # Dictionary for custom command recognition
    command_keywords = {
        "start detection": ["start detection", "begin detection", "activate detection", "turn on detection"],
        "stop detection": ["stop detection", "pause detection", "deactivate detection", "turn off detection"],
        "exit": ["exit", "quit", "close", "shutdown", "terminate"],
        "time": ["what time", "current time", "tell me the time","what is the time", "what is the current time"],
        "date": ["what date", "today's date", "current date","what is the date", "what is the current date"],
        "month": ["what month", "current month", "tell me the month", "what is the month", "what is the current month"],
        "location": ["where am i", "current location", "my location", "what is my location"],
        "help": ["help", "commands", "what can you do", "available commands"],
        "take screenshot": ["take screenshot", "capture screen", "save screen", "screenshot"],
        "battery": ["battery status", "power status", "how much battery"]
    }
    
    while not stop_threads:
        try:
            with sr.Microphone() as source:
                logger.info("Listening for commands...")
                # Adjust for ambient noise to improve recognition
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Set dynamic energy threshold
                recognizer.energy_threshold = 4000
                recognizer.dynamic_energy_threshold = True
                
                # Listen with timeout to prevent blocking
                audio = recognizer.listen(source, timeout=COMMAND_TIMEOUT, phrase_time_limit=5)
                
                # Process audio in try block to catch recognition errors
                try:
                    command = recognizer.recognize_google(audio).lower()
                    logger.info(f"Command recognized: {command}")
                    
                    # Check if command matches any keywords
                    matched_command = None
                    for cmd_type, phrases in command_keywords.items():
                        if any(phrase in command for phrase in phrases):
                            matched_command = cmd_type
                            break
                    
                    # If matched, use the standard command format
                    if matched_command:
                        command_queue.put(matched_command)
                    else:
                        # Add recognized command to queue for processing in main thread
                        command_queue.put(command)
                        
                except sr.UnknownValueError:
                    logger.info("Could not understand the command")
                except sr.RequestError as e:
                    logger.error(f"Could not request results; {e}")
                    speak_text("Speech recognition service unavailable. Check your internet connection.")
                
        except sr.WaitTimeoutError:
            # Timeout occurred, continue the loop
            pass
        except Exception as e:
            logger.error(f"Error in voice recognition: {e}")
            
            # Try to recover if possible
            time.sleep(1)
        
        # Check if we need to retry getting location
        if location_retry_needed and current_location == "Location information unavailable":
            location_retry_needed = False
            # Try again in background
            threading.Thread(target=lambda: globals().update(current_location=get_current_location()), 
                            daemon=True).start()
        
        # Short sleep to prevent CPU overuse
        time.sleep(0.1)

def process_commands():
    """Process commands from the command queue"""
    global detection_active, stop_threads, current_location, detected_objects
    
    while not command_queue.empty():
        try:
            command = command_queue.get_nowait()
            logger.info(f"Processing command: {command}")
            
            # Command processing logic
            if command == "start detection":
                detection_active = True
                speak_text("Starting obstacle detection")
                
            elif command == "stop detection":
                # Clear the detection state and objects when stopping detection
                detection_active = False
                # Clear the detected objects when stopping detection
                detected_objects.clear()
                speak_text("Stopping obstacle detection")
                
            elif command == "exit":
                stop_threads = True
                speak_text("Exiting program")
                
            # Time-related commands
            elif command == "time":
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                response = f"The current time is {current_time}"
                speak_text(response)
                
            elif command == "date":
                current_date = datetime.datetime.now().strftime("%B %d, %Y")
                response = f"Today's date is {current_date}"
                speak_text(response)
                
            elif command == "month":
                current_month = datetime.datetime.now().strftime("%B")
                response = f"The current month is {current_month}"
                speak_text(response)
                
            # Location command
            elif command == "location":
                # Refresh location in background
                threading.Thread(target=lambda: globals().update(current_location=get_current_location()), 
                                daemon=True).start()
                response = f"Your current location is: {current_location or 'being determined'}"
                speak_text(response)
                
            # Help command
            elif command == "help":
                help_text = (
                    "Available commands: start detection, stop detection, what time is it, "
                    "what date is it, what month is it, where am I, take screenshot, or exit"
                )
                speak_text(help_text)
            
            # Screenshot command
            elif command == "take screenshot":
                command_queue.put("__take_screenshot__")  # Special command for webcam thread
                speak_text("Taking a screenshot")
            
            # Battery status (simulated)
            elif command == "battery":
                # In a real implementation, this would check actual battery
                # For now, just use a simulated value
                battery_level = 75  # Simulated battery percentage
                speak_text(f"Battery level is approximately {battery_level} percent")
            
            # Mark command as processed
            command_queue.task_done()
        
        except queue.Empty:
            break
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            command_queue.task_done()  # Make sure to mark as done even on error

def take_screenshot(frame):
    """Save a screenshot of the current frame"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SCREENSHOT_DIR / f"screenshot_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Screenshot saved: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return False

def calculate_object_size(box, frame_width, frame_height):
    """Calculate relative size of object in frame (as a fraction of total area)"""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    # Calculate area of box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    
    # Calculate fraction of frame occupied
    frame_area = frame_width * frame_height
    relative_size = box_area / frame_area
    
    return relative_size, box_height

def estimate_distance(object_height_px, object_class, frame_height, focal_length):
    """
    Estimate distance to object in meters using focal length and object height
    Formula: distance = (actual_height * focal_length) / pixel_height
    """
    # Get reference height for the object class
    actual_height = REFERENCE_HEIGHTS.get(object_class, REFERENCE_HEIGHTS['default'])
    
    # Avoid division by zero
    if object_height_px == 0:
        return float('inf')
    
    # Calculate distance using the formula
    distance = (actual_height * focal_length) / object_height_px
    
    return distance

def calibrate_camera(cap):
    """Estimate focal length based on camera parameters"""
    # Get camera intrinsic parameters if available
    try:
        # For known camera dimensions
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Typical field of view for webcams: 60-70 degrees horizontally
        # Using 65 degrees as an approximation
        horizontal_fov = 65 * (math.pi / 180)  # Convert to radians
        
        # Estimate focal length from field of view
        focal_length = (frame_width / 2) / math.tan(horizontal_fov / 2)
        
        logger.info(f"Estimated focal length: {focal_length:.2f} pixels")
        return focal_length
    
    except Exception as e:
        logger.error(f"Error calibrating camera: {e}")
        # Fallback focal length estimation based on typical webcam
        return 600  # Common approximation for 640x480 webcams

def find_best_path(left_obstacles, center_obstacles, right_obstacles):
    """
    Determine the best available path based on obstacle distribution
    Returns: (path_direction, confidence_level)
    """
    # Calculate "obstacle density" for each region
    left_count = len(left_obstacles)
    center_count = len(center_obstacles)
    right_count = len(right_obstacles)
    
    # Simple path finding - choose path with fewest obstacles
    if left_count <= center_count and left_count <= right_count:
        if left_count == 0:
            return ("left", "clear")
        else:
            return ("left", "least obstructed")
    elif center_count <= left_count and center_count <= right_count:
        if center_count == 0:
            return ("straight ahead", "clear")
        else:
            return ("straight ahead", "least obstructed")
    else:
        if right_count == 0:
            return ("right", "clear")
        else:
            return ("right", "least obstructed")

def webcam_detection_thread():
    """Thread for processing webcam feed and performing object detection"""
    global detection_active, stop_threads, detected_objects, FOCAL_LENGTH, frame_regions
    
    # Initialize YOLO model with the lighter model for better performance
    logger.info(f"Loading YOLOv8n model... This may take a moment.")
    
    try:
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        speak_text("Error loading object detection model. Please check installation and try again.")
        stop_threads = True
        return
    
    # Open webcam
    logger.info("Opening webcam...")
    
    # Try multiple camera indices if first one fails
    cap = None
    for cam_index in range(3):  # Try indices 0, 1, 2
        try:
            cap = cv2.VideoCapture(cam_index)
            if cap.isOpened():
                logger.info(f"Successfully opened camera at index {cam_index}")
                break
        except Exception:
            continue
    
    # Check if webcam opened successfully
    if not cap or not cap.isOpened():
        logger.error("Error: Could not open webcam")
        speak_text("Error: Could not access camera. Please check connections and try again.")
        stop_threads = True
        return
    
    logger.info("Webcam opened successfully!")
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Increased frame size for better detection
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Calibrate camera to estimate focal length
    FOCAL_LENGTH = calibrate_camera(cap)
    
    # Calculate frame regions once
    global frame_regions
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        frame_regions = {
            'left': (0, 0, width // 3, height),
            'center': (width // 3, 0, 2 * width // 3, height),
            'right': (2 * width // 3, 0, width, height)
        }
    else:
        logger.error("Failed to read initial frame")
        speak_text("Failed to read initial frame. Exiting.")
        return
    
    # Get frame dimensions from camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create window for display
    cv2.namedWindow('Navigation Assistant', cv2.WINDOW_NORMAL)
    
    last_feedback_time = 0
    frame_count = 0
    last_frame = None  # Store last successful frame
    
    # Class names from COCO dataset (YOLOv8 default)
    class_names = model.names
    
    # Performance tracking
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # For potential frame drop detection
    consecutive_failures = 0
    
    # Tracking objects between frames
    current_objects = {}
    
    # Dictionary to track when paths were last announced
    path_announcements = {
        "left": 0,
        "straight": 0,
        "right": 0,
        "all_clear": 0
    }
    
    # Path recommendation cooldown in seconds
    PATH_RECOMMENDATION_COOLDOWN = 10.0
    
    # Add at the start of webcam_detection_thread (after variable declarations)
    last_announced = {
        'left': {'object': None, 'still_said': False, 'last_beep': 0},
        'center': {'object': None, 'still_said': False, 'last_beep': 0},
        'right': {'object': None, 'still_said': False, 'last_beep': 0},
        'clear_side': {'side': None, 'said': False}
    }
    
    BEEP_INTERVAL = 2.0  # seconds
    BEEP_FREQ = 800      # Hz (for winsound)
    BEEP_DUR = 150       # ms
    
    while not stop_threads:
        try:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                logger.warning(f"Failed to grab frame (attempt {consecutive_failures})")
                
                # If we've had too many consecutive failures, try to recover
                if consecutive_failures >= 10:
                    logger.error("Too many frame capture failures. Attempting to recover camera...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(0)
                    consecutive_failures = 0
                    continue
                
                # Use last successful frame if available
                if last_frame is not None:
                    frame = last_frame.copy()
                    logger.info("Using last successful frame as fallback")
                else:
                    time.sleep(0.1)
                    continue
            else:
                # Reset consecutive failures counter
                consecutive_failures = 0
                # Save this as the last successful frame
                last_frame = frame.copy()
            
            # Check for special commands (screenshots)
            try:
                # Non-blocking check for screenshot command
                if not command_queue.empty() and command_queue.queue[0] == "__take_screenshot__":
                    command_queue.get()  # Remove from queue
                    success = take_screenshot(frame)
                    if success:
                        speak_text("Screenshot saved")
                    else:
                        speak_text("Failed to save screenshot")
                    command_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing special command: {e}")
            
            # Pre-processing: use BGR (no blur, no color conversion)
            
            # Create a visualization frame (copy of original)
            vis_frame = frame.copy()
            
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Increment frame counter
            frame_count += 1
            fps_counter += 1
            
            # Update FPS calculation every second
            if time.time() - fps_start_time > 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0
            
            # Run detection only on some frames when active
            run_detection = detection_active and (frame_count % SKIP_FRAMES == 0)
            
            # Clear current objects map each detection cycle
            current_objects = {}
            
            if run_detection:
                # Run YOLOv8 inference on the frame with reduced resolution for speed
                small_frame = cv2.resize(frame, (640, 480))
                results = model(small_frame, conf=CONFIDENCE_THRESHOLD)
                
                if results and len(results) > 0:
                    # Get frame dimensions
                    small_height, small_width = small_frame.shape[:2]
                    
                    # Scale factor for mapping back to original frame
                    scale_x = frame_width / small_width
                    scale_y = frame_height / small_height
                    
                    # Track obstacles by position with their relative sizes
                    left_obstacles = []
                    center_obstacles = []
                    right_obstacles = []
                    
                    # Track closest obstacle in each region
                    closest_left = (None, 0, float('inf'))  # (name, size, distance)
                    closest_center = (None, 0, float('inf'))
                    closest_right = (None, 0, float('inf'))
                    
                    # Draw results on visualization frame
                    for r in results:
                        boxes = r.boxes
                        
                        for box in boxes:
                            # Get box coordinates and confidence
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Scale coordinates back to original frame size
                            x1 *= scale_x
                            y1 *= scale_y
                            x2 *= scale_x
                            y2 *= scale_y
                            
                            confidence = box.conf[0].item()
                            class_id = int(box.cls[0].item())
                            
                            # Only process if confidence is above threshold
                            if confidence > CONFIDENCE_THRESHOLD:
                                # Get object class name
                                class_name = class_names[class_id]
                                
                                # Calculate center of the box
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                
                                # Calculate relative size of object
                                rel_size, box_height_px = calculate_object_size(box, frame_width, frame_height)
                                
                                # Calculate distance estimate in meters
                                distance = estimate_distance(box_height_px, class_name, frame_height, FOCAL_LENGTH)
                                
                                # Create object ID based on position and class
                                box_width = x2 - x1
                                obj_id = f"{class_name}_{int(x_center/10)}_{int(y_center/10)}_{int(box_width/10)}"
                                
                                # Store object information
                                current_objects[obj_id] = {
                                    "class": class_name,
                                    "confidence": confidence,
                                    "size": rel_size,
                                    "distance": distance,
                                    "box": (x1, y1, x2, y2),
                                    "center": (x_center, y_center),
                                    "last_seen": time.time()
                                }
                                
                                # Color depends on whether object is a critical obstacle and proximity
                                if class_name in CRITICAL_OBSTACLES and distance < 4.0:
                                    box_color = (0, 0, 255)  # Red for critical close obstacles
                                elif distance < 2.0:
                                    box_color = (0, 0, 255)  # Red for very close obstacles
                                elif distance < 4.0:
                                    box_color = (0, 165, 255)  # Orange for moderately close obstacles
                                else:
                                    box_color = (0, 255, 0)  # Green for distant obstacles
                                
                                # For each region, check overlap
                                if x1 < frame_width // 3 and x2 > 0:
                                    left_obstacles.append((class_name, rel_size, distance))
                                    if distance < closest_left[2]:
                                        closest_left = (class_name, rel_size, distance)
                                if x1 < 2 * frame_width // 3 and x2 > frame_width // 3:
                                    center_obstacles.append((class_name, rel_size, distance))
                                    if distance < closest_center[2]:
                                        closest_center = (class_name, rel_size, distance)
                                if x1 < frame_width and x2 > 2 * frame_width // 3:
                                    right_obstacles.append((class_name, rel_size, distance))
                                    if distance < closest_right[2]:
                                        closest_right = (class_name, rel_size, distance)
                                
                                # Draw bounding box with improved visuals
                                cv2.rectangle(vis_frame, 
                                            (int(x1), int(y1)), 
                                            (int(x2), int(y2)), 
                                            box_color, 2)
                                
                                # Add label with class name, confidence, and distance
                                label = f"{class_name} ({confidence:.2f}): {distance:.1f}m"
                                cv2.putText(vis_frame, label, 
                                            (int(x1), int(y1) - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.5, box_color, 2)
                    
                    # Get current time for cooldown check
                    current_time = time.time()
                    
                    # Update the detected objects with current objects
                    # This keeps track of objects between frames
                    for obj_id, obj_data in current_objects.items():
                        obj_class = obj_data["class"]
                        obj_size = obj_data["size"]
                        obj_distance = obj_data["distance"]
                        
                        # Check if this is a new object or an existing one with significant change
                        if obj_id not in detected_objects:
                            # New object
                            detected_objects[obj_id] = {
                                "class": obj_class, 
                                "size": obj_size,
                                "distance": obj_distance,
                                "first_seen": current_time,
                                "last_announced": 0,  # Never announced
                                "last_seen": current_time
                            }
                        else:
                            # Existing object - update data but preserve announcement timing
                            prev_size = detected_objects[obj_id]["size"] 
                            prev_distance = detected_objects[obj_id]["distance"]
                            
                            # Calculate change as percentage
                            size_change = abs(obj_size - prev_size) / max(prev_size, 0.01)
                            distance_change = abs(obj_distance - prev_distance) / max(prev_distance, 0.01)
                            
                            # Update the entry
                            detected_objects[obj_id]["size"] = obj_size
                            detected_objects[obj_id]["distance"] = obj_distance
                            detected_objects[obj_id]["last_seen"] = current_time
                            
                            # Mark for re-announcement if significant change
                            if size_change > SIZE_CHANGE_THRESHOLD or distance_change > SIZE_CHANGE_THRESHOLD:
                                # If there's significant change, reset the cooldown timer
                                # This ensures important changes are announced
                                if detected_objects[obj_id]["last_announced"] > 0:
                                    # Only reset if it was announced before (avoid instant announcements)
                                    elapsed = current_time - detected_objects[obj_id]["last_announced"]
                                    if elapsed > FEEDBACK_COOLDOWN:
                                        # Only reset if basic cooldown has passed
                                        detected_objects[obj_id]["last_announced"] = 0
                    
                    # Clean up old objects that weren't seen in this frame
                    object_ids = list(detected_objects.keys())
                    for obj_id in object_ids:
                        if obj_id not in current_objects:
                            # Keep track of when we last saw this object
                            time_since_last_seen = current_time - detected_objects[obj_id]["last_seen"]
                            if time_since_last_seen > 2.0:  # If not seen for 2 seconds, remove it
                                del detected_objects[obj_id]
                    
                    # --- Improved region-object mapping and voice output ---
                    # Map: object_name -> set of regions
                    object_regions = {}
                    for region, obstacles in zip(['left', 'center', 'right'], [left_obstacles, center_obstacles, right_obstacles]):
                        for obj in obstacles:
                            name = obj[0]
                            if name not in object_regions:
                                object_regions[name] = set()
                            object_regions[name].add(region)

                    # Track last announced state for each object
                    if not hasattr(webcam_detection_thread, 'last_object_regions'):
                        webcam_detection_thread.last_object_regions = {}
                    last_object_regions = webcam_detection_thread.last_object_regions
                    now = time.time()
                    BEEP_INTERVAL = 2.5
                    BEEP_FREQ = 800
                    BEEP_DUR = 150

                    for obj_name, regions in object_regions.items():
                        regions_sorted = sorted(regions, key=lambda r: ['left','center','right'].index(r))
                        region_str = ' and '.join(regions_sorted)
                        announce_str = f"{obj_name} detected on your {region_str}."
                        # Only announce if regions changed
                        if last_object_regions.get(obj_name) != regions_sorted:
                            speak_text(announce_str)
                            last_object_regions[obj_name] = regions_sorted
                            # Reset beep timer
                            last_announced[obj_name] = {'last_beep': now}
                        else:
                            # Beep if object remains in same region(s)
                            if obj_name not in last_announced:
                                last_announced[obj_name] = {'last_beep': 0}
                            if now - last_announced[obj_name]['last_beep'] > BEEP_INTERVAL:
                                if platform.system() == 'Windows':
                                    winsound.Beep(BEEP_FREQ, BEEP_DUR)
                                else:
                                    os.system('printf "\a"')
                                last_announced[obj_name]['last_beep'] = now
                    # Remove objects that are no longer present
                    for obj_name in list(last_object_regions.keys()):
                        if obj_name not in object_regions:
                            del last_object_regions[obj_name]
                            if obj_name in last_announced:
                                del last_announced[obj_name]

                    # --- Improved overlay logic ---
                    overlay = vis_frame.copy()
                    alpha = 0.25  # Transparency for overlay
                    region_bounds = [
                        (0, frame_width // 3, 'left'),
                        (frame_width // 3, 2 * frame_width // 3, 'center'),
                        (2 * frame_width // 3, frame_width, 'right')
                    ]
                    region_has_object = {
                        'left': bool(left_obstacles),
                        'center': bool(center_obstacles),
                        'right': bool(right_obstacles)
                    }
                    for x1, x2, region in region_bounds:
                        color = (0, 255, 0) if not region_has_object[region] else (0, 0, 255)
                        cv2.rectangle(overlay, (x1, 0), (x2, frame_height), color, -1)
                    cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)

                    # Announce closest object and its distance for each region
                    for region, closest in zip(['left', 'center', 'right'], [closest_left, closest_center, closest_right]):
                        if closest[0]:  # Only if an object is detected in this region
                            obj_name, _, dist_m = closest
                            dist_ft = dist_m * 3.28
                            steps = int(dist_ft / 2.5)
                            dist_str = f"{int(dist_ft)} feet ({steps} steps)"
                            speak_text(f"{obj_name} detected in the {region} at {dist_str}.")

                    # Guidance based on region blockage
                    left_clear = not region_has_object['left']
                    center_clear = not region_has_object['center']
                    right_clear = not region_has_object['right']

                    if not region_has_object['center'] and not region_has_object['left'] and not region_has_object['right']:
                        speak_text("Path is clear. You can move ahead.")
                    elif region_has_object['center']:
                        if left_clear and right_clear:
                            speak_text("Obstacle ahead. You can move left or right.")
                        elif left_clear:
                            speak_text("Obstacle ahead. You can move left.")
                        elif right_clear:
                            speak_text("Obstacle ahead. You can move right.")
                        else:
                            speak_text("Obstacles detected in all directions. Please stop.")
                    elif region_has_object['left'] and not region_has_object['right']:
                        speak_text("Obstacle on your left. You can move right.")
                    elif region_has_object['right'] and not region_has_object['left']:
                        speak_text("Obstacle on your right. You can move left.")
                    elif region_has_object['left'] and region_has_object['right'] and center_clear:
                        speak_text("Obstacles on your left and right. You can move ahead.")
            
            # Draw vertical lines to separate regions
            cv2.line(vis_frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (255, 255, 255), 2)
            cv2.line(vis_frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (255, 255, 255), 2)
            
            # Add labels for the regions
            cv2.putText(vis_frame, "Left", (frame_width // 6 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(vis_frame, "Center", (frame_width // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(vis_frame, "Right", (5 * frame_width // 6 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Draw status information
            status_color = (0, 255, 0) if detection_active else (0, 0, 255)
            status_text = "Detection: ACTIVE" if detection_active else "Detection: INACTIVE"
            cv2.putText(vis_frame, status_text, (10, frame_height - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Add FPS counter
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, frame_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the resulting frame
            cv2.imshow('Navigation Assistant', vis_frame)
            
            # Process commands in main thread to avoid threading issues
            process_commands()
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
                break
            
        except Exception as e:
            logger.error(f"Error in webcam thread: {e}")
            try:
                # Try to continue with next frame
                time.sleep(0.1)
            except:
                break
    
    # Release resources when done
    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass

def main():
    """Main program function"""
    # Register signal handlers for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Welcome message
    logger.info("Starting Navigation Assistant...")
    speak_text("Navigation Assistant starting. Say help for available commands.")
    
    # Create and start threads
    threads = []
    
    # Speech output thread
    speech_thread_obj = threading.Thread(target=speech_thread, daemon=True)
    speech_thread_obj.start()
    threads.append(speech_thread_obj)
    
    # Voice command thread
    voice_thread_obj = threading.Thread(target=voice_command_thread, daemon=True)
    voice_thread_obj.start()
    threads.append(voice_thread_obj)
    
    # Webcam detection thread (runs in main thread to avoid OpenCV issues)
    try:
        webcam_detection_thread()
    except Exception as e:
        logger.error(f"Fatal error in main thread: {e}")
        stop_threads = True
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=1.0)
    
    logger.info("Navigation Assistant stopped")
    print("Navigation Assistant stopped")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        speak_text("A fatal error occurred. The program will exit.")
        sys.exit(1)
# MADE BY : Lucas Raoul, Frederic Vinet, Johan Mons, Malo Gueguen
# What is it doing : 
#    This code has been made for a DJI Tello Edu.
#    It uses the libraries: 
#           "OpenCv" to capture the video feed from the drone
#           "Threading" to execute multiple tasks simultaneously
#           "Djitellopy" to connect the Python code to the drone 
#           "Ultralytics" for the YOLO algorithms, which use AI to detect objects with the camera
#           "Flask" to open the video feed on a web interface

import cv2 
import threading
import time
import os
from djitellopy import Tello
from ultralytics import YOLO
from flask import Flask, Response
from datetime import datetime

# --- Initialize Tello Drone & YOLO ---

photos_dir = "captured_photos"
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# Initialize Tello to control the drone
tello = Tello()
tello.connect()  # Connect to the drone
print(f"Battery: {tello.get_battery()}%")  # Display the battery percentage

# Activate the video feed from the drone's camera
tello.streamon()
frame_read = tello.get_frame_read()

# Activate the detection of mission pads (specific to DJI Tello Edu)
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(0)  # Set detection direction (0: downward sensor, 1: front camera)

# Load the YOLO model for object detection
model = YOLO("yolov8n.pt")

# Define a lock to synchronize access to frames
frame_lock = threading.Lock()

# Flag global pour le cooldown
cooldown = False
cooldown_time = 10  # Time in seconds to ignore captures after detection


def add_ar_elements(frame, tello, detections):
    """
    Adds Augmented Reality (AR) elements to the given frame.
    Includes battery and the number of detected objects.
    """
    # 1. Battery Informations
    battery = tello.get_battery()
    cv2.putText(frame, f"Battery: {battery}%", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Number of objects detected
    num_objects = len(detections)
    cv2.putText(frame, f"Number of Objects: {num_objects}", (10, 110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame

def save_photo(frame):
    """
    Saves a photo with a unique timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(photos_dir, f"person_detected_{timestamp}.jpg")
    with frame_lock:
        cv2.imwrite(filename, frame)
    print(f"[Fly Thread] Photo saved: {filename}")

def cooldown_timer(duration):
    """
    Disables cooldown after a specified duration.
    """
    global cooldown
    time.sleep(duration)
    cooldown = False
    print("[Fly Thread] Cooldown deactivated. Taking another photo.")


# --- Flight and Detection Functions ---

def fly_sequence():
    """
    Handle the flight sequence of the drone and use YOLO to detect a person.
    If a person is detected, the drone performs a flip to the right and lands.
    """
    global cooldown
    print("[Fly Thread] Taking off...")
    tello.takeoff()  # Drone takes off

    # 1) Move forward until pad #4 is detected
    while True:
        pad_id = tello.get_mission_pad_id()
        # 2) Verify which pad it is
        if pad_id == 4:  # Pad #4 detected
            print("[Fly Thread] Pad #4 detected.")
            print("[Fly Thread] 90° rotation...")
            tello.rotate_clockwise(90)
            break
        elif pad_id == 8:  # Pad #4 detected
            print("[Fly Thread] Pad #8 detected.")
            print("[Fly Thread] 180° rotation...")
            tello.rotate_clockwise(180)
            time.sleep(1)
            print("[Fly Thread] Moving 50 cm up...")
            tello.move_up(50)
            time.sleep(1)
            break
        elif pad_id == -1:
            print("[Fly Thread] No pad detected. Moving forward...")
        else:
            print(f"[Fly Thread] Detected pad ID: {pad_id}. Continuing search...")

        tello.move_forward(50)  # Move forward
        time.sleep(1)  # Pause for 1 second
    # 3) Detect a person using YOLO
    while True:
        img = frame_read.frame  # Retrieve the current frame from the video feed
        if img is None:
            continue  # If no frame is available, continue

        # Object detection with YOLO
        results = model.predict(source=img, save=False, conf=0.5)
        detections = []
        person_detected = False

        for result in results[0].boxes.data:
            class_id = int(result[5])  # Detected class index
            class_name = model.names.get(class_id, "Unknown")  # Class name
            confidence = float(result[4])  # Confidence score
            x1, y1, x2, y2 = map(int, result[:4])  # Bounding box coordinates

            if class_name == 'person':
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
                if not cooldown:
                    person_detected = True
                    print("[Fly Thread] Person detected ! Taking a photo...")
                    # Capture and save the photo
                    save_photo(img)
                    # Activate cooldown
                    cooldown = True
                    # Start a thread to deactivate cooldown after a specified time
                    threading.Thread(target=cooldown_timer, args=(cooldown_time,), daemon=True).start()
                    # Perform a flip to the right
                    tello.flip("r")
                    break  # Exit the loop once the person is detected
        if person_detected:
            break 

    # 4) Land the drone
    print("[Fly Thread] Landing...")
    tello.land()

# --- Flask Server for the Video Feed ---

app = Flask(__name__)  # Initialize the Flask application

def gen_frames():
    global cooldown
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        # Object detection with YOLO
        results = model.predict(source=frame, save=False, conf=0.5)
        detections = []
        person_detected = False

        for result in results[0].boxes.data:
            class_id = int(result[5])  # Detected class index
            class_name = model.names.get(class_id, "Unknown")  # Class name
            confidence = float(result[4])  # Confidence score
            x1, y1, x2, y2 = map(int, result[:4])  # Bounding box coordinates

            if class_name == 'person':
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
                if not cooldown:
                    person_detected = True
                    # Capture and save the photo
                    save_photo(frame)
                    # Activate the cooldown
                    cooldown = True
                    # Launch a thread to deactivate the cooldown after a certain time
                    threading.Thread(target=cooldown_timer, args=(cooldown_time,), daemon=True).start()

        # Adding AR elements
        annotated_frame = add_ar_elements(frame, tello, detections)
        # Convert the frame from BGR (used by OpenCV) to RGB (used for browser display)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Encode the annotated frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        # Construct the MJPEG stream with the encoded JPEG frame
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)  # Pause briefly to limit the stream's frame rate

@app.route('/video_feed')
def video_feed():
    """
    Flask route to serve the video feed.
    This endpoint streams the annotated video feed in MJPEG format to the browser.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Main HTML page displaying the live video feed from the Tello drone.
    The page is styled with basic CSS for better presentation.
    """
    return '''
    <html>
    <head>
        <title>Tello Video Feed</title>
        <style>
            body {
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
            }
            h1 {
                color: #333;
                text-align: center;
                margin-top: 30px;
            }
            .video-container {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-top: 20px;
            }
            img {
                border: 2px solid #ccc;
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Flux vidéo Tello avec Détection de Personnes</h1>
        <div class="video-container">
            <img src="/video_feed" width="640" height="480" />
        </div>
    </body>
    </html>
    '''

# --- Launch the Application ---

if __name__ == '__main__':
    try:
        # Thread 1: Manage the flight sequence (fly_sequence)
        fly_thread = threading.Thread(target=fly_sequence, daemon=True)
        fly_thread.start()

        # Main thread: Launch the Flask server
        app.run(host='0.0.0.0', port=5000, debug=False)

    finally:
        # Properly stop the video feed
        tello.streamoff()
        # Attempt to land the drone if still in flight
        try:
            tello.land()
        except:
            pass
        # Release the drone's resources
        tello.end()

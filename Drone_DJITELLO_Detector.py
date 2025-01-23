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
from djitellopy import Tello
from ultralytics import YOLO
from flask import Flask, Response

# --- Initialize Tello Drone & YOLO ---

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

# --- Flight and Detection Functions ---

def fly_sequence():
    """
    Handle the flight sequence of the drone and use YOLO to detect a person.
    If a person is detected, the drone performs a flip to the right and lands.
    """
    print("[Fly Thread] Taking off...")
    tello.takeoff()  # Drone takes off

    # 1) Move forward until pad #4 is detected
    while True:
        pad_id = tello.get_mission_pad_id()
        if pad_id == 4:  # Pad #4 detected
            print("[Fly Thread] Pad #4 detected.")
            # 2) Rotate 90° clockwise
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
    person_detected = False
    while not person_detected:
        img = frame_read.frame  # Retrieve the current frame from the video feed
        if img is None:
            continue  # Skip if no frame is available

        # Perform object detection with YOLO
        results = model.predict(source=img, save=False, conf=0.5)

        # Check detection results to see if a person is in the frame
        for result in results[0].boxes.data:
            class_id = int(result[5])
            class_name = model.names.get(class_id, "Unknown")
            if class_name == 'person':
                person_detected = True
                print("[Fly Thread] Person detected!")
                tello.flip("r")  # Perform a flip to the right
                break

    # 4) Land the drone
    print("[Fly Thread] Landing...")
    tello.land()

# --- Flask Server for the Video Feed ---

app = Flask(__name__)  # Initialize the Flask application

def gen_frames():
    """
    Generate an MJPEG video feed with YOLO detections and annotations.
    """
    while True:
        frame = frame_read.frame
        if frame is None:
            continue

        # Perform object detection with YOLO and annotate the frame
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()  # Draw bounding boxes on the image (BGR format)

        # Convert BGR to RGB for better visualization in the web interface
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Encode the image as a JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Construct the MJPEG feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')

        time.sleep(0.03)  # Pause to limit the feed flow

@app.route('/video_feed')
def video_feed():
    """
    Route that exposes the annotated video feed in MJPEG format.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Main HTML page displaying the video feed with simple styling.
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
        <h1>Tello Video Feed (YOLO)</h1>
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

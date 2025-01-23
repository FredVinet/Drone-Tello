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

# Définir un verrou pour synchroniser l'accès aux frames
frame_lock = threading.Lock()

# Flag global pour le cooldown
cooldown = False
cooldown_time = 10  # Temps en secondes pendant lequel les captures sont ignorées après une détection


def add_ar_elements(frame, tello, detections):
    """
    Ajoute des éléments de Réalité Augmentée (AR) sur le frame fourni.
    Inclut les informations de batterie et altitude, une flèche de direction,
    une zone de suivi, et le nombre d'objets détectés.
    """
    # 1. Informations de Batterie et Altitude
    battery = tello.get_battery()
    altitude = tello.get_height()
    cv2.putText(frame, f"Batterie: {battery}%", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Altitude: {altitude} cm", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Flèche de Direction
    height, width, _ = frame.shape
    arrow_start = (width // 2, height - 50)
    arrow_end = (width // 2, height - 150)
    cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 5)
    cv2.putText(frame, "Direction", (arrow_end[0] + 10, arrow_end[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3. Zone de Suivi
    center_x, center_y = width // 2, height // 2
    box_size = 100
    top_left = (center_x - box_size, center_y - box_size)
    bottom_right = (center_x + box_size, center_y + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(frame, "Zone de Suivi", (top_left[0], top_left[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 4. Nombre d'Objets Détectés
    num_objects = len(detections)
    cv2.putText(frame, f"Objets détectés: {num_objects}", (10, 110),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 5. Labels pour les Objets Détectés
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['box']
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def save_photo(frame):
    """
    Sauvegarde une photo avec un horodatage unique.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(photos_dir, f"person_detected_{timestamp}.jpg")
    with frame_lock:
        cv2.imwrite(filename, frame)
    print(f"[Fly Thread] Photo sauvegardée: {filename}")

def cooldown_timer(duration):
    """
    Désactive le cooldown après une certaine durée.
    """
    global cooldown
    time.sleep(duration)
    cooldown = False
    print("[Fly Thread] Cooldown désactivé. Prêt à capturer de nouvelles photos.")


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
    while True:
        img = frame_read.frame  # Récupère le cadre actuel du flux vidéo
        if img is None:
            continue  # Si aucun cadre n'est disponible, continue

        # Détection d'objets avec YOLO
        results = model.predict(source=img, save=False, conf=0.5)
        detections = []
        person_detected = False

        for result in results[0].boxes.data:
            class_id = int(result[5])  # Indice de classe détectée
            class_name = model.names.get(class_id, "Unknown")  # Nom de la classe
            confidence = float(result[4])  # Score de confiance
            x1, y1, x2, y2 = map(int, result[:4])  # Coordonnées de la bounding box

            if class_name == 'person':
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
                if not cooldown:
                    person_detected = True
                    print("[Fly Thread] Personne détectée ! Capture de la photo...")
                    # Capture et sauvegarde de la photo
                    save_photo(img)
                    # Activer le cooldown
                    cooldown = True
                    # Lancer un thread pour désactiver le cooldown après un certain temps
                    threading.Thread(target=cooldown_timer, args=(cooldown_time,), daemon=True).start()
                    # Effectuer un flip vers la droite
                    tello.flip("r")
                    break  # Sort de la boucle une fois la personne détectée

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

        # Détection d'objets avec YOLO
        results = model.predict(source=frame, save=False, conf=0.5)
        detections = []
        person_detected = False

        for result in results[0].boxes.data:
            class_id = int(result[5])  # Indice de classe détectée
            class_name = model.names.get(class_id, "Unknown")  # Nom de la classe
            confidence = float(result[4])  # Score de confiance
            x1, y1, x2, y2 = map(int, result[:4])  # Coordonnées de la bounding box

            if class_name == 'person':
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
                if not cooldown:
                    person_detected = True
                    # Capture et sauvegarde de la photo
                    save_photo(frame)
                    # Activer le cooldown
                    cooldown = True
                    # Lancer un thread pour désactiver le cooldown après un certain temps
                    threading.Thread(target=cooldown_timer, args=(cooldown_time,), daemon=True).start()

        # Ajout des éléments AR
        annotated_frame = add_ar_elements(frame, tello, detections)

        # Écriture de la frame annotée dans le fichier vidéo
        # video_writer.write(annotated_frame)

        # Conversion de BGR en RGB pour un affichage correct dans le navigateur
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Encodage de l'image annotée en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Construction du flux MJPEG
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.03)  # Pause pour limiter le débit du flux

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

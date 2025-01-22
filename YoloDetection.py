import cv2
import threading
from djitellopy import Tello
from ultralytics import YOLO

# Initialisation du drone
tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

# Chargement du modèle YOLO
model = YOLO("yolov8n.pt")

# Fonction de vol avec obstacles
def fly_sequence():
    print("Tentative de décollage...")
    print("Battery:", tello.get_battery())
    tello.takeoff()

    while True:
        img = frame_read.frame
        if img is None:
            continue

        # Détection avec YOLO
        results = model.predict(source=img, save=False, conf=0.5)

        for result in results[0].boxes.data:
            class_id = int(result[5])  # ID de la classe
            if model.names[class_id] == 'person':  # Si un humain est détecté
                print("Humain détecté.")
                print("Do a flip.")
                tello.flip("r")
            
        tello.land()
        break

def display_video():
    while True:
        img = frame_read.frame
        if img is None:
            continue

        results = model.predict(source=img, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        cv2.imshow("Tello Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

try:
    fly_thread = threading.Thread(target=fly_sequence)
    fly_thread.start()

    display_video()

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")
finally:
    tello.streamoff()
    cv2.destroyAllWindows()

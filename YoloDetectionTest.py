import cv2
import threading
from djitellopy import Tello
from ultralytics import YOLO
import time

# Initialisation du drone
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

# Chargement du modèle YOLO
model = YOLO("yolov8n.pt")

# Fonction d'exploration et de détection
def explore_and_detect():
    print("Tentative de décollage...")
    print(f"Battery: {drone.get_battery()}%")
    drone.takeoff()

    try:
        while True:
            # Capture une image
            img = frame_read.frame
            if img is None:
                continue

            # Détection avec YOLO
            results = model.predict(source=img, save=False, conf=0.5)

            person_detected = False

            for result in results[0].boxes.data:
                box_names = int(result[5])  # Nom de la classe
                if model.names[box_names] == 'person':
                    person_detected = True
                    print(f"Humain détecté ! Réalisation d'un flip.")
                    break

            if person_detected:
                print("Personne détectée")
                drone.flip("r")
                time.sleep(3)

            if not person_detected:
                print("Aucune personne détectée. Exploration en cours...")
                drone.move_forward(50)
                time.sleep(2)
                drone.rotate_clockwise(90)
                time.sleep(2)

    except Exception as e:
        print(f"Erreur rencontrée : {e}")
    finally:
        print("Atterrissage...")
        drone.land()

# Fonction d'affichage vidéo
def display_video():
    while True:
        img = frame_read.frame
        if img is None:
            continue

        # Annoter les résultats YOLO sur l'image
        results = model.predict(source=img, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Afficher la vidéo
        cv2.imshow("Tello Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

try:
    explore_thread = threading.Thread(target=explore_and_detect, daemon=True)  # Le thread tourne en arrière-plan
    explore_thread.start()

    display_video()

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")
finally:
    drone.streamoff()
    cv2.destroyAllWindows()
    drone.land()

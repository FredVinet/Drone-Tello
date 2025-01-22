import cv2
import threading
from djitellopy import Tello
from ultralytics import YOLO
import time

# Initialisation du drone
tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

# Chargement du modèle YOLO
model = YOLO("yolov8n.pt")  # Assurez-vous que yolov8n.pt est présent

# Fonction de mouvements du drone
def fly_sequence():
    print("Tentative de décollage...")
    print("Battery:", tello.get_battery())
    tello.takeoff()
    time.sleep(2)
    tello.move_back(30)
    time.sleep(2)
    tello.rotate_clockwise(360)
    time.sleep(5)
    tello.move_down(20)
    time.sleep(2)
    tello.move_back(30)
    time.sleep(2)
    tello.land()

# Fonction pour l'affichage vidéo
def display_video():
    while True:
        # Récupération de la frame pour YOLO
        img = frame_read.frame
        if img is None:
            continue

        # Détection avec YOLO
        results = model.predict(source=img, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Affichage de la détection sur OpenCV
        cv2.imshow("Tello Stream", annotated_frame)

        # Sortie avec 'q' (fermer la fenêtre)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

try:
    # Créer un thread pour les déplacements du drone
    fly_thread = threading.Thread(target=fly_sequence)
    fly_thread.start()

    # Affichage vidéo dans le thread principal
    display_video()

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")
finally:
    tello.streamoff()
    cv2.destroyAllWindows()

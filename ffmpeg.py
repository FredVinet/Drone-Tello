import cv2
import threading
from djitellopy import Tello
from ultralytics import YOLO
import time

# Initialisation du drone
drone = Tello()
drone.connect()
print(f"Battery: {drone.get_battery()}%")

# Active la réception du flux vidéo
drone.streamon()

# URL du flux vidéo (via FFmpeg)
video_stream_url = "udp://0.0.0.0:11111"

# Chargement du modèle YOLO
model = YOLO("yolov8n.pt")

# Fonction d'exploration et de détection
# def explore_and_detect():
#     print("Tentative de décollage...")
#     drone.takeoff()

#     try:
#         cap = cv2.VideoCapture(video_stream_url, cv2.CAP_FFMPEG)
#         if not cap.isOpened():
#             print("[Error] Impossible d'ouvrir le flux vidéo.")
#             return

#         while True:
#             # Capture une image
#             ret, img = cap.read()
#             if not ret or img is None:
#                 print("[Warning] Aucune image reçue. Vérifiez la connexion au drone.")
#                 continue

#             # Détection avec YOLO
#             results = model.predict(source=img, save=False, conf=0.5)

#             person_detected = False
#             for result in results[0].boxes.data:
#                 box_names = int(result[5])  # Nom de la classe
#                 if model.names[box_names] == 'person':
#                     person_detected = True
#                     print(f"Humain détecté ! Réalisation d'un flip.")
#                     break

#             if person_detected:
#                 print("Personne détectée")
#                 drone.flip("r")
#                 time.sleep(3)

#             if not person_detected:
#                 print("Aucune personne détectée. Exploration en cours...")
#                 drone.move_forward(50)
#                 time.sleep(2)
#                 drone.rotate_clockwise(90)
#                 time.sleep(2)

#     except Exception as e:
#         print(f"Erreur rencontrée : {e}")
#     finally:
#         print("Atterrissage...")
#         drone.land()

# Fonction d'affichage vidéo
def display_video():
    cap = cv2.VideoCapture(video_stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire le tampon

    if not cap.isOpened():
        print("[Error] Impossible d'ouvrir le flux vidéo.")
        return

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            print("[Warning] Aucune image reçue.")
            continue

        # Option : Désactiver YOLO sur certaines frames (par exemple, 1 sur 5)
        if int(time.time() * 10) % 5 == 0:
            results = model.predict(source=img, save=False, conf=0.5, classes=[0])
            annotated_frame = results[0].plot()
        else:
            annotated_frame = img  # Pas de traitement YOLO

        # Afficher l'image
        cv2.imshow("Tello Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Lancer les threads pour l'exploration et l'affichage
try:
    # explore_thread = threading.Thread(target=explore_and_detect, daemon=True)  # Le thread tourne en arrière-plan
    # explore_thread.start()

    display_video()

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")
finally:
    drone.streamoff()
    cv2.destroyAllWindows()
    # drone.land()

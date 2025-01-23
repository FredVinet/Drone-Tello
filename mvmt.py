import cv2
import threading
import time
from djitellopy import Tello
from ultralytics import YOLO

# --- Initialisation du drone ---
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Active la réception du flux vidéo
tello.streamon()
frame_read = tello.get_frame_read()

# Active la détection de pads (Tello Edu)
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(0)  # Vers le bas

# --- Chargement du modèle YOLO ---
model = YOLO("yolov8n.pt")

def fly_sequence():
    print("[Fly Thread] Décollage...")
    tello.takeoff()

    # 1) Avancer jusqu’à détecter le pad #4
    while True:
        pad_id = tello.get_mission_pad_id()
        print("Pad ID =", pad_id)  # Debug
        if pad_id == 4:
            print("[Fly Thread] Pad #4 détecté. Arrêt de la progression.")
            break
        else:
            print("[Fly Thread] Pas encore de pad #4. J'avance...")
            tello.move_forward(50)
            time.sleep(1)

    # 2) Tourner à 90° (rotation horaire)
    print("[Fly Thread] Rotation de 90°...")
    tello.rotate_clockwise(90)

    # 3) Lancer la détection YOLO
    person_detected = False
    while not person_detected:
        img = frame_read.frame
        if img is None:
            continue

        results = model.predict(source=img, save=False, conf=0.5)
        # Vérifie s'il y a un humain dans la détection
        for result in results[0].boxes.data:
            class_id = int(result[5])  # indice de classe
            class_name = model.names.get(class_id, "Unknown")
            if class_name == 'person':
                person_detected = True
                print("[Fly Thread] Personne détectée ! Flip latéral...")
                tello.flip("r")  # flip sur la droite
                break

    # 4) Atterrissage
    print("[Fly Thread] Atterrissage...")
    tello.land()

def display_video():
    while True:
        img = frame_read.frame
        if img is None:
            continue

        # On peut annoter pour l'affichage
        results = model.predict(source=img, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        cv2.imshow("Tello Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

try:
    # Thread 1 : gestion du vol (fly_sequence)
    fly_thread = threading.Thread(target=fly_sequence, daemon=True)
    fly_thread.start()

    # Thread 2 : affichage du flux vidéo
    display_video()

except KeyboardInterrupt:
    print("[Main] Interruption par l'utilisateur.")

finally:
    # Arrêt propre
    tello.streamoff()
    cv2.destroyAllWindows()
    # Au cas où le drone serait encore en vol
    try:
        tello.land()
    except:
        pass

import cv2
import threading
from djitellopy import Tello
from ultralytics import YOLO
import time


tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

model = YOLO("yolov8n.pt")

def explore_and_detect():
    print("Tentative de décollage...")
    print(f"Battery: {tello.get_battery()}%")
    tello.takeoff()

    try:
        while True:
            img = frame_read.frame
            if img is None:
                continue

            results = model.predict(source=img, save=False, conf=0.5)
            person_detected = False

            for result in results[0].boxes.data:
                class_id = int(result[5])
                confidence = result[4]
                if model.names[class_id] == 'person' and confidence > 0.6:
                    person_detected = True
                    print(f"Humain détecté avec une confiance de {confidence:.2f} ! Réalisation d'un flip.")
                    tello.flip("r")
                    time.sleep(3)
                    break

            if not person_detected:
                print("Aucune personne détectée. Exploration en cours...")
                tello.move_forward(50)
                time.sleep(2)
                tello.rotate_clockwise(90)
                time.sleep(2)

    except Exception as e:
        print(f"Erreur rencontrée : {e}")
    finally:
        print("Atterrissage...")
        tello.land()

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
    explore_thread = threading.Thread(target=explore_and_detect)
    explore_thread.start()

    display_video()

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")
finally:
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.land()

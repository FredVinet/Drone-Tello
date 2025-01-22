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

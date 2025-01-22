from djitellopy import Tello
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Initialisation du drone
tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

# Chargement du modèle YOLOv8
model = YOLO("yolov8n.pt")  # Assurez-vous que yolov8n.pt est présent

try:
    plt.ion()  # Mode interactif
    fig, ax = plt.subplots()

    while True:
        # Lecture de la frame actuelle
        img = frame_read.frame
        if img is None:
            continue

        # Détection avec YOLO
        results = model.predict(source=img, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Affichage via Matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.pause(0.001)

        # Sortie avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interruption par l'utilisateur.")

finally:
    tello.streamoff()
    plt.close()

import cv2
from djitellopy import Tello

# --- Initialisation du drone ---
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Active la réception du flux vidéo
tello.streamon()

# Capture du flux vidéo avec OpenCV et FFmpeg
video_stream_url = "udp://0.0.0.0:11111"


def display_video():
    cap = cv2.VideoCapture(video_stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[Error] Impossible d'ouvrir le flux vidéo.")
        return

    print("[Info] Flux vidéo en cours d'affichage. Appuyez sur 'q' pour quitter.")
    while True:
        ret, img = cap.read()
        if not ret or img is None:
            print("[Warning] Aucune image reçue. Vérifiez la connexion au drone.")
            continue

        # Affiche le flux vidéo
        cv2.imshow("Tello Stream", img)

        # Quitte si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


try:
    display_video()

except KeyboardInterrupt:
    print("[Main] Interruption par l'utilisateur.")

finally:
    # Arrête le flux vidéo proprement
    tello.streamoff()
    cv2.destroyAllWindows()
    print("[Main] Programme terminé.")

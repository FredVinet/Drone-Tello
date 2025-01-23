from djitellopy import Tello
import cv2
import time
from flask import Flask, Response
from ultralytics import YOLO
import os
import threading
from datetime import datetime

# --- INITIALISATION DU DRONE & YOLO ---

# Définir le répertoire où les photos seront sauvegardées
photos_dir = "captured_photos"
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)

# Initialisation de l'objet Tello pour contrôler le drone
tello = Tello()
tello.connect()  # Connexion au drone
print(f"Batterie: {tello.get_battery()}%")  # Affichage du niveau de batterie

# Activation de la réception du flux vidéo du drone
tello.streamon()
frame_read = tello.get_frame_read()

# Chargement du modèle YOLO pour la détection d'objets
model = YOLO("yolov8n.pt")  # Assurez-vous que le modèle est téléchargé

# --- INITIALISATION DES THREADS ---

# Définir un verrou pour synchroniser l'accès aux frames
frame_lock = threading.Lock()

# Flag global pour le cooldown
cooldown = False
cooldown_time = 10  # Temps en secondes pendant lequel les captures sont ignorées après une détection

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

# --- FONCTIONS DE VOL & DÉTECTION ---

def fly_sequence():
    """
    Gère la séquence de vol du drone et utilise YOLO pour détecter une personne.
    Si une personne est détectée, capture une photo, effectue un flip à droite et atterrit.
    """
    global cooldown
    print("[Fly Thread] Décollage...")
    tello.takeoff()  # Décollage du drone

    # 1) Avancer jusqu'à détecter le pad #4
    while True:
        pad_id = tello.get_mission_pad_id()  # Récupère l'ID du pad détecté
        print("Pad ID =", pad_id)  # Affichage pour le débogage
        if pad_id == 4:
            print("[Fly Thread] Pad #4 détecté. Arrêt de la progression.")
            break
        else:
            print("[Fly Thread] Pas encore de pad #4. J'avance...")
            tello.move_forward(50)  # Avance de 50 cm
            time.sleep(1)  # Pause d'une seconde

    # 2) Tourner de 90° dans le sens horaire
    print("[Fly Thread] Rotation de 90°...")
    tello.rotate_clockwise(90)

    # 3) Boucle pour détecter une personne
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
            break  # Quitter la boucle principale pour atterrir

    # 4) Atterrissage du drone
    print("[Fly Thread] Atterrissage...")
    tello.land()

# --- SERVEUR FLASK POUR LE FLUX VIDÉO ---

app = Flask(__name__)  # Initialisation de l'application Flask

def add_ar_elements(frame, tello, detections):
    """
    Ajoute des éléments de Réalité Augmentée (AR) sur le frame fourni.
    Inclut les informations de batterie et altitude, une flèche de direction,
    une zone de suivi, et le nombre d'objets détectés.
    """
    # 1. Informations de Batterie et Altitude
    battery = tello.get_battery()
    altitude = tello.get_height()
    cv2.putText(frame, f"Batterie: {battery}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Altitude: {altitude} cm", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 2. Flèche de Direction
    height, width, _ = frame.shape
    arrow_start = (width // 2, height - 50)
    arrow_end = (width // 2, height - 150)
    cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 5)
    cv2.putText(frame, "Direction", (arrow_end[0] + 10, arrow_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 3. Zone de Suivi
    center_x, center_y = width // 2, height // 2
    box_size = 100
    top_left = (center_x - box_size, center_y - box_size)
    bottom_right = (center_x + box_size, center_y + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(frame, "Zone de Suivi", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 4. Nombre d'Objets Détectés
    num_objects = len(detections)
    cv2.putText(frame, f"Objets détectés: {num_objects}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 5. Labels pour les Objets Détectés
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        x1, y1, x2, y2 = detection['box']
        cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

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
        video_writer.write(annotated_frame)

        # Conversion de BGR en RGB pour un affichage correct dans le navigateur
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Encodage de l'image annotée en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        # Construction du flux MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        time.sleep(0.03)  # Pause pour limiter le débit du flux

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Flux vidéo Tello avec Détection</title>
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

# --- LANCEMENT DE L'APPLICATION ---

if __name__ == '__main__':
    try:
        # Thread 1 : Gestion de la séquence de vol (fly_sequence)
        fly_thread = threading.Thread(target=fly_sequence, daemon=True)
        fly_thread.start()

        # Thread principal : Lancement du serveur Flask
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Arrêt propre du flux vidéo
        tello.streamoff()
        # Libération de VideoWriter
        video_writer.release()
        # Tentative d'atterrissage si le drone est encore en vol
        try:
            tello.land()
        except:
            pass
        # Libération des ressources du drone
        tello.end()

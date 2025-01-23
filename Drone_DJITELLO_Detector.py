import cv2 
import threading
import time
from djitellopy import Tello
from ultralytics import YOLO
from flask import Flask, Response

# --- INITIALISATION DU DRONE & YOLO ---

# Initialisation de l'objet Tello pour contrôler le drone
tello = Tello()
tello.connect()  # Connexion au drone
print(f"Batterie: {tello.get_battery()}%")  # Affichage du niveau de batterie

# Activation de la réception du flux vidéo du drone
tello.streamon()
frame_read = tello.get_frame_read()

# Activation de la détection des pads (pour Tello Edu)
tello.enable_mission_pads()
tello.set_mission_pad_detection_direction(0)  # Détection vers le bas

# Chargement du modèle YOLO pour la détection d'objets
model = YOLO("yolov8n.pt")

# --- FONCTIONS DE VOL & DÉTECTION ---

def fly_sequence():
    """
    Gère la séquence de vol du drone et utilise YOLO pour détecter une personne.
    Si une personne est détectée, effectue un flip à droite et atterrit.
    """
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
    person_detected = False
    while not person_detected:
        img = frame_read.frame  # Récupère le cadre actuel du flux vidéo
        if img is None:
            continue  # Si aucun cadre n'est disponible, continue

        # Utilise YOLO pour prédire les objets dans l'image
        results = model.predict(source=img, save=False, conf=0.5)
        
        # Parcourt les résultats pour vérifier la présence d'une personne
        for result in results[0].boxes.data:
            class_id = int(result[5])  # Indice de classe détectée
            class_name = model.names.get(class_id, "Unknown")  # Nom de la classe
            if class_name == 'person':
                person_detected = True
                print("[Fly Thread] Personne détectée ! Flip latéral...")
                tello.flip("r")  # Effectue un flip vers la droite
                break  # Sort de la boucle une fois la personne détectée

    # 4) Atterrissage du drone
    print("[Fly Thread] Atterrissage...")
    tello.land()

# --- SERVEUR FLASK POUR LE FLUX VIDÉO ---

app = Flask(__name__)  # Initialisation de l'application Flask

def gen_frames():
    """
    Génère un flux vidéo au format MJPEG avec les annotations de détection YOLO.
    """
    while True:
        frame = frame_read.frame  # Récupère le cadre actuel du flux vidéo
        if frame is None:
            continue  # Si aucun cadre n'est disponible, continue

        # Détection d'objets avec YOLO pour annoter le cadre
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()  # Dessine les bounding boxes sur l'image (format BGR)

        # Conversion de BGR en RGB pour un affichage correct dans le navigateur
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Encodage de l'image annotée en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue  # Si l'encodage échoue, continue

        # Construction du flux MJPEG
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')

        time.sleep(0.03)  # Pause pour limiter le débit du flux

@app.route('/video_feed')
def video_feed():
    """
    Route qui expose le flux vidéo au format MJPEG annoté.
    """
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """
    Page HTML principale affichant le flux vidéo avec un style simple.
    """
    return '''
    <html>
    <head>
        <title>Flux vidéo Tello</title>
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
        <h1>Flux vidéo Tello (YOLO)</h1>
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
        # Tentative d'atterrissage si le drone est encore en vol
        try:
            tello.land()
        except:
            pass
        # Libération des ressources du drone
        tello.end()

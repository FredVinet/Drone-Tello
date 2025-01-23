from djitellopy import Tello
import cv2
import time
from flask import Flask, Response

tello = Tello()
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

app = Flask(__name__)

def gen_frames():
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes()
               + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Tello Stream</title>
    </head>
    <body>
        <h1>Flux vidéo Tello</h1>
        <img src="/video_feed" width="640" height="480" />
    </body>
    </html>
    '''

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # On s'assure que le flux est bien arrêté et que le port est libéré
        tello.streamoff()
        tello.end()

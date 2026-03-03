from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

app = Flask(__name__)

model = tf.keras.models.load_model("asl_model_final.keras")

class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

prediction_buffer = deque(maxlen=5)

def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("Camera not opened")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        x1, y1 = 200, 100
        x2, y2 = 450, 350

        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 40), 2)

        roi = frame[y1:y2, x1:x2]

        try:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = cv2.resize(roi, (128, 128))
            img = img / 255.0
            img = np.reshape(img, (1, 128, 128, 3))

            prediction = model.predict(img, verbose=0)
            index = np.argmax(prediction)

            prediction_buffer.append(index)

            if len(prediction_buffer) == prediction_buffer.maxlen:
                index = max(set(prediction_buffer), key=prediction_buffer.count)

            label = class_names[index]

        except:
            label = "Detecting..."

        cv2.putText(frame, f"{label}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (40, 40, 40),
                    2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
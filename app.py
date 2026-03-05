from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
import os
from keras.models import load_model

app = Flask(__name__)

# Lazy model loading
model = None

def get_model():
    global model
    if model is None:
        model = load_model("asl_model_final.keras", compile=False, safe_mode=False)
    return model


labels = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z","nothing"
]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/live")
def live():
    return render_template("live.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]

        encoded = data.split(",")[1]
        img_bytes = base64.b64decode(encoded)

        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img,axis=0)

        prediction = get_model().predict(img, verbose=0)
        label = labels[int(np.argmax(prediction))]

        return jsonify({"prediction": label})

    except Exception as e:
        print("Prediction error:", e)
    return jsonify({"prediction": "Error"})
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT",10000))
    app.run(host="0.0.0.0", port=port)
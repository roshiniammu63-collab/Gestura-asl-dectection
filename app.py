from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
import cv2
from keras.models import load_model

app = Flask(__name__)

# Load trained ASL model
model = None

def get_model():
    global model
    if model is None:
        from keras.models import load_model
        model = load_model("asl_model_final.keras", compile=False)
    return model

# Alphabet labels
labels = [
"A","B","C","D","E","F","G","H","I","J",
"K","L","M","N","O","P","Q","R","S","T",
"U","V","W","X","Y","Z",
"del","nothing","space"
]


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- LIVE PAGE ----------------
@app.route("/live")
def live():
    return render_template("live.html")


# ---------------- ABOUT PAGE ----------------
@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    image_data = data["image"].split(",")[1]

    image_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize to model input
    img = cv2.resize(img, (128, 128))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype("float32") / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0) 

    prediction = get_model().predict(img)

    confidence = np.max(prediction)

    class_index = np.argmax(prediction)

    # Only show prediction if confidence is good
    if confidence > 0.75:
        letter = labels[class_index]
    else:
        letter = "--"

    return jsonify({"prediction": letter})


# ---------------- RUN APP ----------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
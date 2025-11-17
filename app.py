from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import os
import time

app = Flask(__name__)

# Load your model once
MODEL_PATH = "models/model.h5"
model = load_model(MODEL_PATH)

# Label dictionary
label_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder to store temporary/debug images
DEBUG_FOLDER = "static/debug"
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# ---------------------- Routes ---------------------- #

@app.route('/')
def home():
    return render_template('index.html')


# Predict from uploaded file
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    timestamp = int(time.time())
    debug_path = os.path.join(DEBUG_FOLDER, f"upload_{timestamp}.png")
    
    # Save for debugging
    file.save(debug_path)

    # Load and preprocess image
    img = Image.open(debug_path).convert("L").resize((48,48))
    img_array = np.array(img).reshape(1,48,48,1)/255.0

    # Predict
    prediction = model.predict(img_array)
    emotion_index = int(np.argmax(prediction))
    emotion = label_dict[emotion_index]

    # Return HTML
    return render_template('index.html', prediction=emotion, image_path=debug_path)


# Capture from webcam
@app.route('/capture', methods=['POST'])
def capture():
    cap = cv2.VideoCapture(0)
    
    # Warm up camera
    for _ in range(20):
        ret, frame = cap.read()

    cap.release()

    if not ret:
        return "Failed to capture from webcam", 500

    timestamp = int(time.time())
    debug_path = os.path.join(DEBUG_FOLDER, f"capture_{timestamp}.jpg")
    cv2.imwrite(debug_path, frame)

    # Convert to grayscale and detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return render_template('index.html', prediction="No face detected", image_path=debug_path)

    # Use the first detected face
    x, y, w, h = faces[0]
    face_img = gray[y:y+h, x:x+w]

    # Preprocess for model
    img = Image.fromarray(face_img).resize((48,48))
    img_array = np.array(img).reshape(1,48,48,1)/255.0

    prediction = model.predict(img_array)
    emotion_index = int(np.argmax(prediction))
    emotion = label_dict[emotion_index]

    return render_template('index.html', prediction=emotion, image_path=debug_path)


# Optional JSON API route
@app.route('/predict_api', methods=['POST'])
def predict_api():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400

    img = Image.open(file.stream).convert("L").resize((48,48))
    img_array = np.array(img).reshape(1,48,48,1)/255.0

    prediction = model.predict(img_array)
    emotion_index = int(np.argmax(prediction))
    emotion = label_dict[emotion_index]

    return jsonify({"emotion": emotion})


# ---------------------- Run ---------------------- #
if __name__ == '__main__':
    app.run(debug=True)

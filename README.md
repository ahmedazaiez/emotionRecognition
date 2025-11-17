# Emotion Recognition Web App

A Flask-based web application for real-time and uploaded image emotion recognition using a pre-trained Keras model.

---

## Features
- Upload an image to detect emotion.
- Capture image from webcam to detect emotion in real-time.
- Displays the captured image along with the predicted emotion.

---

## Prerequisites
- Python 3.10+
- All dependencies listed in `requirements.txt`.

---

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedazaiez/emotionRecognition.git
   cd emotionRecognition
2. Install dependencies:

```bash
   pip install -r requirements.txt

```

3. Place your model:

- Add your model.h5 file in the models/ folder.

- Example path: models/model.h5.

- you will have to run the emotionrecognition.ipynb file

4. Run the app:
```bash
python3 app.py
```

5. Open your browser at: http://127.0.0.1:5000 



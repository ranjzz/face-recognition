# face-recognition
pip install tensorflow keras opencv-python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained CNN model for face recognition
model = load_model('path/to/face_recognition_model.h5')

# Function to preprocess the input image
def preprocess_image(image, target_size=(150, 150)):
    image = cv2.resize(image, target_size)
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to perform face recognition
def recognize_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = preprocess_image(face)
        prediction = model.predict(face)[0]

        label = "Recognized" if prediction > 0.5 else "Not Recognized"
        color = (0, 255, 0) if label == "Recognized" else (0, 0, 255)

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    return image

# Example usage
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = recognize_face(frame)
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime

# -------- Load model ----------
model = tf.keras.models.load_model("best_balanced_noaug.keras")

class_names = ['clearskin', 'dark_spots', 'puffy_eyes', 'wrinkles']

# -------- Load Haar Cascade ---
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# -------- Load image ----------
img_path = "test.jpg"
image = cv2.imread(img_path)

if image is None:
    print("❌ test.jpg not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:
    print("❌ No face detected")
    exit()

(x, y, w, h) = faces[0]
face = image[y:y+h, x:x+w]

face = cv2.resize(face, (224, 224))
face = np.expand_dims(face, axis=0)
face = preprocess_input(face)

# -------- Prediction ----------
preds = model.predict(face)[0]
pred_class = class_names[np.argmax(preds)]
confidence = float(np.max(preds) * 100)

# -------- Draw annotation ----
cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
cv2.putText(
    image,
    f"{pred_class} ({confidence:.1f}%)",
    (x, y-10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0,255,0),
    2
)

# -------- Save output image ---
cv2.imwrite("output.jpg", image)

# -------- Save CSV log --------
with open("predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "image", "predicted_class", "confidence"])
    writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        img_path,
        pred_class,
        f"{confidence:.2f}"
    ])

print("✅ output.jpg saved")
print("✅ predictions.csv saved")

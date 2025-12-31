import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------- Load trained model --------
model = tf.keras.models.load_model("best_balanced_noaug.keras")

# ⚠️ Class order same rakho jo training me thi
class_names = ['clearskin', 'dark_spots', 'puffy_eyes', 'wrinkles']

# -------- Load Haar Cascade ----------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# -------- Load test image ------------
img_path = "test.jpg"
image = cv2.imread(img_path)

if image is None:
    print("❌ test.jpg not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -------- Face detection -------------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=5
)

if len(faces) == 0:
    print("❌ No face detected")
    exit()

# -------- Use first detected face ----
(x, y, w, h) = faces[0]
face = image[y:y+h, x:x+w]

face = cv2.resize(face, (224, 224))
face = np.expand_dims(face, axis=0)
face = preprocess_input(face)

# -------- Prediction -----------------
preds = model.predict(face)[0]
pred_class = class_names[np.argmax(preds)]
confidence = np.max(preds) * 100

print(f"Predicted Class: {pred_class}")
print(f"Confidence     : {confidence:.2f}%")

# -------- Draw bounding box ----------
cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.putText(
    image,
    f"{pred_class} ({confidence:.1f}%)",
    (x, y-10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imshow("DermalScan - Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

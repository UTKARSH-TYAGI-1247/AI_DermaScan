import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# -------- Page config ----------
st.set_page_config(
    page_title="DermalScan",
    page_icon="🧴",
    layout="wide"
)
st.title("🧴 DermalScan – Facial Skin Aging Detection")
st.markdown("""
<div style="background-color:#5b8df0;padding:20px;border-radius:10px">
<p style="font-size:16px">
Instant AI-driven insights into facial skin aging — smart, fast, and effortless.<br>
</p>
</div>
""", unsafe_allow_html=True)

if "prediction_logs" not in st.session_state:
    st.session_state.prediction_logs = []


# -------- Load model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_balanced_noaug.keras")

model = load_model()

class_names = ['clearskin', 'dark_spots', 'puffy_eyes', 'wrinkles']

# -------- Load Haar Cascade ---
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
st.subheader("📸 Select Input Method")
input_mode = st.radio("Choose input source:", ["Upload Image", "Use Webcam"])

def generate_pdf_report(df, image_array, final_label, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "DermalScan – Skin Condition Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, "Educational use only. Not a medical diagnosis.")

    # Prediction summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 120, "Prediction Summary")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 145, f"Predicted Class: {final_label}")
    c.drawString(50, height - 165, f"Confidence: {confidence:.2f}%")
    c.drawString(50, height - 185, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Table header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 230, "Prediction Logs")

    y = height - 260
    c.setFont("Helvetica-Bold", 10)
    headers = ["Image Name", "Timestamp", "Prediction", "Confidence (%)"]
    x_positions = [50, 180, 330, 450]

    for header, x in zip(headers, x_positions):
        c.drawString(x, y, header)

    y -= 15
    c.setFont("Helvetica", 10)

    for _, row in df.iterrows():
        c.drawString(x_positions[0], y, str(row["Image Name"]))
        c.drawString(x_positions[1], y, str(row["Timestamp"]))
        c.drawString(x_positions[2], y, str(row["Prediction"]))
        c.drawString(x_positions[3], y, str(row["Confidence (%)"]))
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50

    c.showPage()
    c.save()

    buffer.seek(0)
    return buffer


# -------- Upload image --------
uploaded_file = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )
elif input_mode == "Use Webcam":
    uploaded_file = st.camera_input("Take a picture")

st.divider()


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected. Please upload a clear frontal face image.")
    else:
        primary_preds = None
        primary_pred_class = None
        primary_confidence = None
        for i, (x, y, w, h) in enumerate(faces, start=1):
            face = img_array[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (224, 224))
            face_resized = np.expand_dims(face_resized, axis=0)
            face_resized = preprocess_input(face_resized)
            preds = model.predict(face_resized)[0]
            pred_class = class_names[np.argmax(preds)]
            confidence = np.max(preds) * 100
            if i == 1:
                primary_preds = preds
                primary_pred_class = pred_class
                primary_confidence = confidence
            if input_mode == "Upload Image":
                image_name = f"{uploaded_file.name}_face_{i}"
            else:
                image_name = f"webcam_face_{i}.jpg"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.prediction_logs.append({
                "Image Name": image_name,
                "Timestamp": timestamp,
                "Prediction": pred_class,
                "Confidence (%)": round(confidence, 2)
            })
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(
                img_array,
                f"{pred_class} ({confidence:.1f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )
        st.divider()
        st.subheader("📊 Prediction Confidence Distribution")
        percentages = {
            "Clear Skin": float(primary_preds[class_names.index("clearskin")] * 100),
            "Dark Spots": float(primary_preds[class_names.index("dark_spots")] * 100),
            "Puffy Eyes": float(primary_preds[class_names.index("puffy_eyes")] * 100),
            "Wrinkles": float(primary_preds[class_names.index("wrinkles")] * 100),
        }
        st.bar_chart(percentages)
        st.subheader("🔍 Detailed Confidence Levels")
        for label, value in percentages.items():
            st.write(f"{label}: {value:.2f}%")
            st.progress(value / 100)
        display_img = cv2.resize(img_array, (800, int(img_array.shape[0] * 800 / img_array.shape[1])))
        st.image(display_img, caption="Prediction Result", width=800)
        st.success(f"Predicted (Primary Face): **{primary_pred_class}**")
        st.info(f"Confidence: **{primary_confidence:.2f}%**")
        st.divider()
        st.subheader("📋 Prediction Logs")
        if len(st.session_state.prediction_logs) > 0:
            df = pd.DataFrame(st.session_state.prediction_logs)
            st.dataframe(df, use_container_width=True)
            # -------- PDF DOWNLOAD BUTTON (YAHI AAYEGA) --------
            pdf_buffer = generate_pdf_report(
                df=df,
                image_array=img_array,
                final_label=primary_pred_class,
                confidence=primary_confidence
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="DermalScan_Report.pdf",
                mime="application/pdf"
            )
        else:
            st.info("No predictions yet.")

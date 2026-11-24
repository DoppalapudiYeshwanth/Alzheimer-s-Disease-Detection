import os
import tempfile
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Alzheimer's MRI Detection", page_icon="🧠", layout="centered")
st.title("🧠 Alzheimer's Disease Detection (YOLOv11)")
st.write("Upload a brain MRI scan to detect and classify dementia stages.")


@st.cache_resource
def load_model():
    return YOLO("models/best.pt")


model = load_model()

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Match the notebook's grayscale preprocessing step
    gray_image = image.convert("L")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    if st.button("Analyze MRI Scan", type="primary"):
        with st.spinner("Running YOLOv11 inference..."):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                gray_image.save(tmp.name)
                results = model.predict(source=tmp.name, conf=0.25)
                os.remove(tmp.name)

            annotated_bgr = results[0].plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            with col2:
                st.subheader("Detection Result")
                st.image(annotated_rgb, use_container_width=True)

            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success("Detections found:")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    st.write(f"- **{label}** (Confidence: `{conf:.2%}`)")
            else:
                st.warning("No dementia markers detected above the confidence threshold.")
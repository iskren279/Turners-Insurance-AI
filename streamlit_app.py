
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Turners Insurance AI",
    page_icon="🚗",
    layout="centered"
)

PROJECT_ROOT = Path(__file__).resolve().parent
TWINCAR_SOURCE = PROJECT_ROOT / "vehicle-brand-test" / "src"

sys.path.insert(0, str(TWINCAR_SOURCE))

from twincar.inference import load_bundle, predict_single

MODEL_REPO = "twincar-group2/twincar-classifier"
MODEL_FILE = "efficientnet_b3_stanford300_augv2_best.pt"


@st.cache_resource(show_spinner=False)
def load_models():
    checkpoint = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    twin_car_bundle = load_bundle(
        Path(checkpoint),
        device="cpu",
        image_size=300
    )

    yolo_model = YOLO("yolo11n.pt")

    return twin_car_bundle, yolo_model


def detect_vehicle_type(image_path, yolo_model):
    predictions = yolo_model(str(image_path))

    for prediction in predictions:
        for detection in prediction.boxes:
            class_id = int(detection.cls[0])
            label = yolo_model.names[class_id]

            if label in ["car", "truck", "bus", "motorcycle"]:
                return label.capitalize()

    return "Vehicle type not detected"


def detect_colour(image_path):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((300, 300))

    pixels = list(image.getdata())

    if not pixels:
        return "Unknown"

    average = tuple(
        sum(pixel[i] for pixel in pixels) // len(pixels)
        for i in range(3)
    )

    r, g, b = average

    if max(r, g, b) - min(r, g, b) < 20:
        if r < 70:
            return "Black"
        elif r > 190:
            return "White"
        return "Grey"

    if r > g * 1.3 and r > b * 1.3:
        return "Red"

    if g > r * 1.2 and g > b * 1.2:
        return "Green"

    if b > r * 1.2 and b > g * 1.1:
        return "Blue"

    if r > 150 and g > 100 and b < 100:
        return "Yellow"

    return "Unknown"


def recognise_vehicle(image_path, twin_car_bundle):
    image = Image.open(image_path).convert("RGB")

    prediction = predict_single(
        image,
        twin_car_bundle,
        top_k=3,
        source_path=str(image_path)
    )

    if prediction.error:
        raise RuntimeError(prediction.error)

    top_predictions = [
        {
            "label": item.label,
            "confidence": round(item.confidence * 100, 2)
        }
        for item in prediction.top_k
    ]

    return {
        "prediction": prediction.prediction,
        "make": prediction.make or "Unknown",
        "model": prediction.model or "Unknown",
        "year": prediction.year or "Unknown",
        "confidence": round(prediction.confidence * 100, 2),
        "top_predictions": top_predictions
    }


# --------------------------------------------------
# APPLICATION INTERFACE
# --------------------------------------------------

st.title("🚗 Turners Insurance AI")
st.subheader("AI-powered vehicle recognition")

st.write(
    "Upload a vehicle photograph to identify its "
    "manufacturer, model, year and vehicle type."
)

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded vehicle image",
        use_container_width=True
    )

    if st.button("Analyse Vehicle", type="primary"):
        temp_path = None

        try:
            with st.spinner("Loading AI models and analysing image..."):
                twin_car_bundle, yolo_model = load_models()

                suffix = Path(uploaded_file.name).suffix or ".jpg"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = Path(temp_file.name)

                with Image.open(temp_path) as image:
                    image.verify()

                vehicle = recognise_vehicle(
                    temp_path,
                    twin_car_bundle
                )

                vehicle_type = detect_vehicle_type(
                    temp_path,
                    yolo_model
                )

                colour = detect_colour(temp_path)

            st.success("Vehicle analysis completed!")

            st.divider()
            st.subheader("Vehicle identification")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Manufacturer", vehicle["make"])
                st.metric("Model", vehicle["model"])
                st.metric("Year", vehicle["year"])

            with col2:
                st.metric("Vehicle type", vehicle_type)
                st.metric("Image colour estimate", colour)
                st.metric(
                    "Model confidence",
                    f'{vehicle["confidence"]}%'
                )

            st.subheader("Top 3 predictions")

            for index, item in enumerate(
                vehicle["top_predictions"],
                start=1
            ):
                st.write(
                    f'{index}. {item["label"]} '
                    f'— {item["confidence"]}%'
                )

            st.caption(
                "AI predictions are estimates and should be "
                "independently verified before insurance decisions."
            )

        except Exception:
            logging.exception("Vehicle analysis failed")
            st.error(
                "Unable to analyse this image. "
                "Please check the image and try again."
            )

        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

st.divider()
st.caption("Turners Insurance AI | Vehicle Recognition Project")

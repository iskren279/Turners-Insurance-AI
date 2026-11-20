
import os
import sys
import uuid
import logging
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "jfif"}

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# LOAD TWINCAR MODEL
# --------------------------------------------------

TWINCAR_SOURCE = PROJECT_ROOT / "vehicle-brand-test" / "src"

sys.path.insert(0, str(TWINCAR_SOURCE))

from twincar.inference import load_bundle, predict_single

MODEL_REPO = "twincar-group2/twincar-classifier"

MODEL_FILE = "efficientnet_b3_stanford300_augv2_best.pt"

logging.info("Downloading or locating TwinCar model...")

checkpoint_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

logging.info("Loading TwinCar model on CPU...")

twin_car_bundle = load_bundle(
    Path(checkpoint_path),
    device="cpu",
    image_size=300
)

logging.info("TwinCar model loaded successfully.")

# --------------------------------------------------
# LOAD YOLO VEHICLE DETECTION MODEL
# --------------------------------------------------

logging.info("Loading YOLO vehicle detection model...")

yolo_model = YOLO("yolo11n.pt")

logging.info("YOLO model loaded successfully.")

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def detect_vehicle_type(image_path):
    """
    Detect general vehicle type using YOLO.
    """

    predictions = yolo_model(str(image_path))

    vehicle_types = []

    for prediction in predictions:
        for detection in prediction.boxes:
            class_id = int(detection.cls[0])
            label = yolo_model.names[class_id]

            if label in ["car", "truck", "bus", "motorcycle"]:
                vehicle_types.append(label)

    if vehicle_types:
        return vehicle_types[0].capitalize()

    return "Vehicle type not detected"


def detect_colour(image_path):
    """
    Estimate the average image colour.

    This is an image-level estimate, not a reliable
    measurement of the vehicle's paint colour.
    """

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


def recognise_vehicle(image_path):
    """
    Use TwinCar to predict vehicle make, model and year.
    """

    image = Image.open(image_path).convert("RGB")

    prediction = predict_single(
        image,
        twin_car_bundle,
        top_k=3,
        source_path=str(image_path)
    )

    if prediction.error:
        raise RuntimeError(prediction.error)

    top_predictions = []

    for item in prediction.top_k:
        top_predictions.append({
            "label": item.label,
            "confidence": round(item.confidence * 100, 2)
        })

    return {
        "prediction": prediction.prediction,
        "make": prediction.make or "Unknown",
        "model": prediction.model or "Unknown",
        "year": prediction.year or "Unknown",
        "confidence": round(prediction.confidence * 100, 2),
        "top_predictions": top_predictions
    }


# --------------------------------------------------
# MAIN ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None

    if request.method == "POST":

        image = request.files.get("image")

        if not image or image.filename == "":
            error = "Please select an image."

        elif not allowed_file(image.filename):
            error = "Please upload a JPG, JPEG or PNG image."

        else:

            original_filename = secure_filename(image.filename)

            # Give each uploaded image a unique filename.
            filename = f"{uuid.uuid4().hex}_{original_filename}"

            image_path = UPLOAD_FOLDER / filename

            try:

                image.save(str(image_path))

                # Check that the uploaded file is a valid image.
                with Image.open(image_path) as uploaded_image:
                    uploaded_image.verify()

                # Analyse the image using TwinCar.
                vehicle_details = recognise_vehicle(image_path)

                # Detect the general vehicle type using YOLO.
                vehicle_type = detect_vehicle_type(image_path)

                # Estimate average image colour.
                colour = detect_colour(image_path)

                # Prepare results for the HTML template.
                result = {
                    "type": vehicle_type,
                    "brands": [
                        vehicle_details["make"]
                    ] if vehicle_details["make"] != "Unknown" else [],

                    "colour": colour,

                    "prediction": vehicle_details["prediction"],
                    "make": vehicle_details["make"],
                    "model": vehicle_details["model"],
                    "year": vehicle_details["year"],
                    "confidence": vehicle_details["confidence"],
                    "top_predictions": vehicle_details["top_predictions"]
                }

            except Exception:
                app.logger.exception("Vehicle image analysis failed")

                error = (
                    "Unable to analyse this image. "
                    "Please try another photograph."
                )

    return render_template(
        "index.html",
        result=result,
        error=error
    )


# --------------------------------------------------
# APPLICATION START
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)

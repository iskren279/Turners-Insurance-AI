import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")

client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        image = request.files["image"]

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(image_path)

        with open(image_path, "rb") as f:
            image_data = f.read()

        analysis = client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS
            ]
        )

        tags = []

        if analysis.tags:
            for tag in analysis.tags.list:
                tags.append(tag.name)

        vehicle_type = "Unknown"
        brands = []
        colour = "Unknown"

        for tag in tags:
            tag_lower = tag.lower()

            if tag_lower in [
                "car",
                "suv",
                "truck",
                "van",
                "vehicle",
                "mid-size car",
                "luxury vehicle"
            ]:
                vehicle_type = tag

            if tag_lower in [
                "chevrolet",
                "lexus",
                "chrysler",
                "toyota",
                "ford",
                "bmw",
                "mercedes",
                "audi",
                "honda",
                "nissan",
                "volkswagen"
            ]:
                brands.append(tag)

            if tag_lower in [
                "silver",
                "black",
                "white",
                "red",
                "blue",
                "grey"
            ]:
                colour = tag

        result = {
            "type": vehicle_type,
            "brands": brands,
            "colour": colour
        }

    return render_template(
        "index.html",
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
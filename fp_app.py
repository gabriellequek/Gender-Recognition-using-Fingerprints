
import base64
import io
import traceback
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
import model_inference_vgg19

app = Flask(__name__)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route("/")
def index():
    return render_template("fp_preds.html")
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPG, BMP, or TIFF."}), 400
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
        pil_image.load()
    except UnidentifiedImageError:
        return jsonify({"error": "Could not read this file as an image."}), 400
    try:
        result = model_inference_vgg19.run_inference(pil_image)
    except FileNotFoundError:
        return jsonify({
            "error": "Model weights not found. Set the FP_MODEL_WEIGHTS environment "
                     "variable to the path of your .npz weights file."
        }), 500
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    return jsonify({
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 1),
        "input_image": "data:image/png;base64," + base64.b64encode(result["input_png"]).decode("ascii"),
        "heatmap_image": "data:image/png;base64," + base64.b64encode(result["heatmap_png"]).decode("ascii"),
    })
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
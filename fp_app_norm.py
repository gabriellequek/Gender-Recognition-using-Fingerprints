import base64
import io
import os

from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
M0 = 128
Var0 = 10000

app1 = Flask(__name__)
app1.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


# ── Normalisation logic (unchanged from your script) ──────────────────────────
def process_channel(channel_arr: np.ndarray) -> np.ndarray:
    """Apply variance-normalisation contrast stretch to a single 2-D channel."""
    M = np.mean(channel_arr)
    Var = np.var(channel_arr)

    if Var == 0:  # flat channel – return as-is
        return channel_arr.astype(np.float64)

    temp = np.sqrt(Var0 * (channel_arr - M) ** 2 / Var)

    mask = np.zeros_like(channel_arr, dtype=np.float64)
    mask[channel_arr > M] = 1
    mask[channel_arr < M] = -1

    return M0 + mask * temp


def process_image(img: Image.Image) -> Image.Image:
    """
    Process a PIL image in RGB.
    Grayscale / palette images are converted to RGB first.
    Each colour channel is processed independently.
    """
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float64)  # shape: (H, W, 3)

    out = np.zeros_like(arr)
    for c in range(3):  # R, G, B
        out[:, :, c] = process_channel(arr[:, :, c])

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def pil_to_data_url(img: Image.Image, ext: str, dpi=None) -> str:
    """Encode a PIL image as a base64 data URL so the front end can set it as an <img src> directly."""
    save_format = "JPEG" if ext in {".jpg", ".jpeg"} else Image.registered_extensions().get(ext, "PNG")
    buffer = io.BytesIO()
    save_kwargs = {}
    if dpi:
        save_kwargs["dpi"] = dpi
    if ext in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = 100
        save_kwargs["subsampling"] = 0  # keep chroma detail
    img.save(buffer, format=save_format, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = MIME_BY_EXT.get(ext, "image/png")
    return f"data:{mime};base64,{encoded}"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app1.route("/", methods=["GET"])
def index():
    return render_template("fp_norm.html")


@app1.route("/normalize", methods=["POST"])
def normalize():
    file = request.files.get("image")

    if file is None or file.filename == "":
        return jsonify({"error": "Please choose an image file first."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        }), 400

    ext = os.path.splitext(file.filename)[1].lower()

    try:
        with Image.open(file.stream) as img:
            dpi = img.info.get("dpi")
            input_data_url = pil_to_data_url(img.convert("RGB"), ext, dpi)
            result = process_image(img)
            normalized_data_url = pil_to_data_url(result, ext, dpi)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    return jsonify({
        "input_image": input_data_url,
        "normalized_image": normalized_data_url,
    })


if __name__ == "__main__":
    app1.run(host="0.0.0.0", port=5001, debug=False)
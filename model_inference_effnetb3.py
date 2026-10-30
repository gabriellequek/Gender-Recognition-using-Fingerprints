import io

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt
from scipy.special import expit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEIGHTS_PATH = r"effnetb3_npmodel2.npz"

SIZE_W = 480
SIZE_H = 640
PAD_FILL = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["F", "M"]

# EfficientNet-B3 stage config: (expand_ratio, kernel, stride, in_ch, out_ch, num_layers)
STAGE_CONFIGS = [
    (1, 3, 1, 40, 24, 2),
    (6, 3, 2, 24, 32, 3),
    (6, 5, 2, 32, 48, 3),
    (6, 3, 2, 48, 96, 5),
    (6, 5, 1, 96, 136, 5),
    (6, 5, 2, 136, 232, 6),
    (6, 3, 1, 232, 384, 2),
]

# Lazily loaded once per process (weights don't change between requests).
_WEIGHTS_CACHE = None


def get_weights():
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is None:
        _WEIGHTS_CACHE = load_weights(WEIGHTS_PATH)
    return _WEIGHTS_CACHE


def load_weights(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Core NumPy ops (unchanged from the original script)
# ---------------------------------------------------------------------------

def conv2d(x, w, b=None, stride=1, padding=0):
    N, C, H, W = x.shape
    C_out, _, kH, kW = w.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    H_p, W_p = x.shape[2], x.shape[3]
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    s0, s1, s2, s3 = x.strides
    shape = (N, C, out_h, out_w, kH, kW)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    patches = as_strided(x, shape=shape, strides=strides)
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, out_h * out_w, C * kH * kW)

    w_flat = w.reshape(C_out, C * kH * kW).T
    out = patches @ w_flat
    if b is not None:
        out = out + b
    out = out.transpose(0, 2, 1).reshape(N, C_out, out_h, out_w)
    return out


def depthwise_conv2d(x, w, stride=1, padding=0):
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    H_p, W_p = x.shape[2], x.shape[3]
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    s0, s1, s2, s3 = x.strides
    shape = (N, C, out_h, out_w, kH, kW)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    patches = as_strided(x, shape=shape, strides=strides)
    w_ = w.reshape(C, kH, kW)
    return np.einsum("nchwkl,ckl->nchw", patches, w_)


def batchnorm2d(x, weight, bias, running_mean, running_var, eps=1e-5):
    scale = weight / np.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return x * scale.reshape(1, -1, 1, 1) + shift.reshape(1, -1, 1, 1)


def silu(x):
    return x * expit(x)


def sigmoid(x):
    return expit(x)


def relu(x):
    return np.maximum(x, 0)


def squeeze_excite(x, fc1_w, fc1_b, fc2_w, fc2_b):
    scale = x.mean(axis=(2, 3), keepdims=True)
    scale = conv2d(scale, fc1_w, fc1_b, stride=1, padding=0)
    scale = silu(scale)
    scale = conv2d(scale, fc2_w, fc2_b, stride=1, padding=0)
    scale = sigmoid(scale)
    return x * scale


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# MBConv block & forward pass (unchanged)
# ---------------------------------------------------------------------------

def mbconv_block(x, weights, prefix, expand_ratio, kernel, stride, in_ch, out_ch):
    identity = x
    li = 0

    if expand_ratio != 1:
        x = conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], None, stride=1, padding=0)
        x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                         weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])
        x = silu(x)
        li += 1

    pad = kernel // 2
    x = depthwise_conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], stride=stride, padding=pad)
    x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                     weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])
    x = silu(x)
    li += 1

    x = squeeze_excite(x, weights[f"{prefix}.block.{li}.fc1.weight"], weights[f"{prefix}.block.{li}.fc1.bias"],
                        weights[f"{prefix}.block.{li}.fc2.weight"], weights[f"{prefix}.block.{li}.fc2.bias"])
    li += 1

    x = conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], None, stride=1, padding=0)
    x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                     weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])

    if stride == 1 and in_ch == out_ch:
        x = x + identity
    return x


def classifier_forward(pooled, weights):
    z1 = linear(pooled, weights["classifier.0.weight"], weights["classifier.0.bias"])
    a1 = relu(z1)
    z2 = linear(a1, weights["classifier.3.weight"], weights["classifier.3.bias"])
    a2 = relu(z2)
    logits = linear(a2, weights["classifier.6.weight"], weights["classifier.6.bias"])
    cache = (z1, z2)
    return logits, cache


def classifier_backward(class_idx, weights, cache):
    z1, z2 = cache
    d_a2 = weights["classifier.6.weight"][class_idx]
    d_z2 = d_a2 * (z2[0] > 0)
    d_a1 = d_z2 @ weights["classifier.3.weight"]
    d_z1 = d_a1 * (z1[0] > 0)
    d_pooled = d_z1 @ weights["classifier.0.weight"]
    return d_pooled


def forward(x, weights, return_features=False):
    x = conv2d(x, weights["features.0.0.weight"], None, stride=2, padding=1)
    x = batchnorm2d(x, weights["features.0.1.weight"], weights["features.0.1.bias"],
                     weights["features.0.1.running_mean"], weights["features.0.1.running_var"])
    x = silu(x)

    for stage_idx, (expand_ratio, kernel, stride, in_ch, out_ch, num_layers) in enumerate(STAGE_CONFIGS, start=1):
        for layer_i in range(num_layers):
            block_in = in_ch if layer_i == 0 else out_ch
            block_stride = stride if layer_i == 0 else 1
            prefix = f"features.{stage_idx}.{layer_i}"
            x = mbconv_block(x, weights, prefix, expand_ratio, kernel, block_stride, block_in, out_ch)

    x = conv2d(x, weights["features.8.0.weight"], None, stride=1, padding=0)
    x = batchnorm2d(x, weights["features.8.1.weight"], weights["features.8.1.bias"],
                     weights["features.8.1.running_mean"], weights["features.8.1.running_var"])
    x = silu(x)

    features = x

    x = features.mean(axis=(2, 3), keepdims=True)
    x = x.reshape(x.shape[0], -1)

    logits, cache = classifier_forward(x, weights)

    if return_features:
        return logits, features, cache
    return logits


def compute_gradcam(features, weights, class_idx, cache):
    alpha = classifier_backward(class_idx, weights, cache)
    fmap = features[0]
    cam = np.einsum("chw,c->hw", fmap, alpha)
    cam = relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ---------------------------------------------------------------------------
# Preprocessing (unchanged)
# ---------------------------------------------------------------------------

def get_padding(img_w, img_h):
    h_padding = (SIZE_W - img_w) / 2
    v_padding = (SIZE_H - img_h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    return int(l_pad), int(t_pad), int(r_pad), int(b_pad)


def center_pad(img, path=""):
    l, t, r, b = get_padding(img.width, img.height)
    if min(l, t, r, b) < 0:
        scale = min(SIZE_W / img.width, SIZE_H / img.height)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.BILINEAR)
        l, t, r, b = get_padding(img.width, img.height)

    new_w = img.width + l + r
    new_h = img.height + t + b
    canvas = Image.new("RGB", (new_w, new_h), (PAD_FILL, PAD_FILL, PAD_FILL))
    canvas.paste(img, (l, t))
    return canvas


# ---------------------------------------------------------------------------
# Web-facing entry point
# ---------------------------------------------------------------------------

def _png_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()

def center_crop(image, target_size):
    target_w, target_h = target_size
    w, h = image.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))

def run_inference(pil_image, alpha=0.45):
    """
    Takes a PIL image (already opened from the uploaded file), runs the
    same prediction + Grad-CAM pipeline as show_gradcam() in the original
    script, and returns a JSON-serializable dict instead of plotting:

        {
            "prediction": "F" | "M",
            "confidence": float (0-100),
            "input_png": bytes,     # padded input, as the model actually sees it
            "heatmap_png": bytes,   # Grad-CAM overlay
        }
    """
    weights = get_weights()

    img = pil_image.convert("RGB")
    padded = center_pad(img)

    arr = np.asarray(padded, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    logits, features, cache = forward(arr, weights, return_features=True)
    probs = softmax(logits)
    pred = int(probs[0].argmax())
    pred_gender = CLASS_NAMES[pred]
    confidence = float(probs[0, pred] * 100)

    cam = compute_gradcam(features, weights, pred, cache)

    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize(padded.size, Image.BILINEAR)
    cam_resized = np.asarray(cam_img, dtype=np.float32) / 255.0

    heatmap_rgb = plt.get_cmap("jet")(cam_resized)[..., :3]
    base_rgb = np.asarray(padded, dtype=np.float32) / 255.0
    overlay = np.clip((1 - alpha) * base_rgb + alpha * heatmap_rgb, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_img = center_crop(overlay_img, img.size)

    return {
        "prediction": pred_gender,
        "confidence": confidence,
        "input_png": _png_bytes(img),
        "heatmap_png": _png_bytes(overlay_img),
    }
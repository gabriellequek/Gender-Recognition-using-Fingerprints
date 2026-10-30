import io
import os

import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # no display backend needed on a server
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Separate env var from the EfficientNet module so both models can be
# configured independently (e.g. if app.py ever needs to serve both).
WEIGHTS_PATH = f"vgg19_weights.npz"

SIZE_W = 480
SIZE_H = 640
PAD_FILL = 255
POOL_SIZE = (5, 5)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["F", "M"]

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
       512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

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
# Core NumPy ops
# ---------------------------------------------------------------------------

def conv2d(x, w, b, stride=1, padding=1):
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
    out = out + b
    out = out.transpose(0, 2, 1).reshape(N, C_out, out_h, out_w)
    return out


def relu(x):
    return np.maximum(x, 0)


def maxpool2d(x, kernel=2, stride=2):
    N, C, H, W = x.shape
    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1
    s0, s1, s2, s3 = x.strides
    shape = (N, C, out_h, out_w, kernel, kernel)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    patches = as_strided(x, shape=shape, strides=strides)
    return patches.max(axis=(4, 5))


def _adaptive_pool_bounds(in_size, out_size):
    starts = (np.arange(out_size) * in_size) // out_size
    ends = ((np.arange(out_size) + 1) * in_size + out_size - 1) // out_size
    return starts, ends


def adaptive_avgpool2d(x, output_size):
    N, C, H, W = x.shape
    out_h, out_w = output_size
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    h_starts, h_ends = _adaptive_pool_bounds(H, out_h)
    w_starts, w_ends = _adaptive_pool_bounds(W, out_w)
    for i in range(out_h):
        for j in range(out_w):
            region = x[:, :, h_starts[i]:h_ends[i], w_starts[j]:w_ends[j]]
            out[:, :, i, j] = region.mean(axis=(2, 3))
    return out


def adaptive_avgpool2d_backward(grad_output, input_shape, output_size):
    """Distribute each pooled gradient uniformly back over the input region
    that produced it -- the adjoint of adaptive_avgpool2d, needed to backprop
    a class score through the pooling layer for Grad-CAM."""
    N, C, H, W = input_shape
    out_h, out_w = output_size
    grad_input = np.zeros((N, C, H, W), dtype=grad_output.dtype)
    h_starts, h_ends = _adaptive_pool_bounds(H, out_h)
    w_starts, w_ends = _adaptive_pool_bounds(W, out_w)
    for i in range(out_h):
        for j in range(out_w):
            h0, h1 = h_starts[i], h_ends[i]
            w0, w1 = w_starts[j], w_ends[j]
            area = (h1 - h0) * (w1 - w0)
            grad_input[:, :, h0:h1, w0:w1] += grad_output[:, :, i:i + 1, j:j + 1] / area
    return grad_input


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Classifier head (Linear-ReLU x3, then Linear -> logits) + Grad-CAM backward
# ---------------------------------------------------------------------------

def classifier_forward(pooled_flat, weights):
    z1 = linear(pooled_flat, weights["classifier.0.weight"], weights["classifier.0.bias"])
    a1 = relu(z1)
    z2 = linear(a1, weights["classifier.3.weight"], weights["classifier.3.bias"])
    a2 = relu(z2)
    z3 = linear(a2, weights["classifier.6.weight"], weights["classifier.6.bias"])
    a3 = relu(z3)
    logits = linear(a3, weights["classifier.9.weight"], weights["classifier.9.bias"])
    cache = (z1, z2, z3)
    return logits, cache


def classifier_backward(class_idx, weights, cache):
    z1, z2, z3 = cache
    d_a3 = weights["classifier.9.weight"][class_idx]
    d_z3 = d_a3 * (z3[0] > 0)
    d_a2 = d_z3 @ weights["classifier.6.weight"]
    d_z2 = d_a2 * (z2[0] > 0)
    d_a1 = d_z2 @ weights["classifier.3.weight"]
    d_z1 = d_a1 * (z1[0] > 0)
    d_pooled_flat = d_z1 @ weights["classifier.0.weight"]
    return d_pooled_flat


def compute_gradcam(features, weights, class_idx, cache):
    """Grad-CAM at the last conv block, backpropagated through the 5x5
    adaptive average pool (unlike a global 1x1 pool, this pool mixes several
    spatial locations per output cell, so the pooling gradient has to be
    spread back across each cell's input region rather than just averaged)."""
    d_pooled_flat = classifier_backward(class_idx, weights, cache)
    C = features.shape[1]
    grad_pooled = d_pooled_flat.reshape(1, C, *POOL_SIZE)
    grad_map = adaptive_avgpool2d_backward(grad_pooled, features.shape, POOL_SIZE)

    alpha = grad_map[0].mean(axis=(1, 2))  # (C,) channel importance weights
    fmap = features[0]
    cam = np.einsum("chw,c->hw", fmap, alpha)
    cam = relu(cam)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward(x, weights, return_features=False):
    idx = 0
    for v in CFG:
        if v == 'M':
            x = maxpool2d(x, 2, 2)
            idx += 1
        else:
            w = weights[f"features.{idx}.weight"]
            b = weights[f"features.{idx}.bias"]
            x = conv2d(x, w, b, stride=1, padding=1)
            x = relu(x)
            idx += 2

    features = x  # last conv block output, before adaptive pooling

    pooled = adaptive_avgpool2d(x, POOL_SIZE)
    pooled_flat = pooled.reshape(pooled.shape[0], -1)

    logits, cache = classifier_forward(pooled_flat, weights)

    if return_features:
        return logits, features, cache
    return logits


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_padding(img_w, img_h):
    h_padding = (SIZE_W - img_w) / 2
    v_padding = (SIZE_H - img_h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    return int(l_pad), int(t_pad), int(r_pad), int(b_pad)


def center_pad(img):
    """Returns (canvas, used_size) -- used_size is the size img was actually
    at when it got pasted onto the canvas (after any downscale), so callers
    can crop back to the right dimensions later instead of the original,
    pre-downscale size."""
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
    return canvas, img.size


def center_crop(image, target_size):
    target_w, target_h = target_size
    w, h = image.size
    left = (w - target_w) // 2
    top = (h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


# ---------------------------------------------------------------------------
# Web-facing entry point
# ---------------------------------------------------------------------------

def _png_bytes(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def run_inference(pil_image, alpha=0.45):
    """
    Takes a PIL image (already opened from the uploaded file), runs the
    VGG-19 prediction + Grad-CAM pipeline, and returns a JSON-serializable
    dict with the same shape as the EfficientNet-B3 module:

        {
            "prediction": "F" | "M",
            "confidence": float (0-100),
            "input_png": bytes,     # padded input, as the model actually sees it
            "heatmap_png": bytes,   # Grad-CAM overlay
        }
    """
    weights = get_weights()

    img = pil_image.convert("RGB")
    padded, used_size = center_pad(img)

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
    overlay_img = center_crop(overlay_img, used_size)

    return {
        "prediction": pred_gender,
        "confidence": confidence,
        "input_png": _png_bytes(img),
        "heatmap_png": _png_bytes(overlay_img),
    }
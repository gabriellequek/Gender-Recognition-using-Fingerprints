'''
For running EfficientNetB3 on COMET on entire test dataset 
Displays confusion matrix, accuracy and F1 score
'''


import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sn


WEIGHTS_PATH = "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/effnetb3_npmodel.npz"
test_csv_path = "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Documents/gender_test_set.csv"

SD302_dict = {
    "P": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/P/500/plain/png/",
    "R": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/R/500/slap-segmented/png/",
    "S": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/S/500/slap-segmented/png/"
}

SIZE_W = 480
SIZE_H = 640
PAD_FILL = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["F", "M"]

# EfficientNet-B3 stage config: (expand_ratio, kernel, stride, in_ch, out_ch, num_layers)
# derived from torchvision's MBConvConfig scaling with width_mult=1.2, depth_mult=1.4
STAGE_CONFIGS = [
    (1, 3, 1, 40, 24, 2),
    (6, 3, 2, 24, 32, 3),
    (6, 5, 2, 32, 48, 3),
    (6, 3, 2, 48, 96, 5),
    (6, 5, 1, 96, 136, 5),
    (6, 5, 2, 136, 232, 6),
    (6, 3, 1, 232, 384, 2),
]


df = pd.read_csv(test_csv_path)
valid_categories = ("P", "R", "S")
df = df[df.iloc[:, 2].isin(valid_categories)].reset_index(drop=True)
df.head()


# ## Weight Loading


def load_weights(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ## Core NumPy Ops
# Im2col-based conv, depthwise conv, batchnorm (inference-mode), SiLU/sigmoid activations, squeeze-excitation, linear, softmax.


def conv2d(x, w, b=None, stride=1, padding=0):
    """Standard conv, groups=1. Used for stem, 1x1 expand/project/head convs."""
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
    """Grouped conv where groups == channels. w shape: (C, 1, kH, kW), no bias (BN follows)."""
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
    patches = as_strided(x, shape=shape, strides=strides)  # (N,C,out_h,out_w,kH,kW)
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


def squeeze_excite(x, fc1_w, fc1_b, fc2_w, fc2_b):
    scale = x.mean(axis=(2, 3), keepdims=True)          # AdaptiveAvgPool2d(1)
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


# ## MBConv Block & Forward Pass


def mbconv_block(x, weights, prefix, expand_ratio, kernel, stride, in_ch, out_ch):
    identity = x
    li = 0  # layer index within block.*

    if expand_ratio != 1:
        x = conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], None, stride=1, padding=0)
        x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                         weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])
        x = silu(x)
        li += 1

    # depthwise
    pad = kernel // 2
    x = depthwise_conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], stride=stride, padding=pad)
    x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                     weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])
    x = silu(x)
    li += 1

    # squeeze-excitation
    x = squeeze_excite(x, weights[f"{prefix}.block.{li}.fc1.weight"], weights[f"{prefix}.block.{li}.fc1.bias"],
                        weights[f"{prefix}.block.{li}.fc2.weight"], weights[f"{prefix}.block.{li}.fc2.bias"])
    li += 1

    # project (no activation)
    x = conv2d(x, weights[f"{prefix}.block.{li}.0.weight"], None, stride=1, padding=0)
    x = batchnorm2d(x, weights[f"{prefix}.block.{li}.1.weight"], weights[f"{prefix}.block.{li}.1.bias"],
                     weights[f"{prefix}.block.{li}.1.running_mean"], weights[f"{prefix}.block.{li}.1.running_var"])

    if stride == 1 and in_ch == out_ch:
        x = x + identity
    return x


def forward(x, weights):
    # stem: features.0
    x = conv2d(x, weights["features.0.0.weight"], None, stride=2, padding=1)
    x = batchnorm2d(x, weights["features.0.1.weight"], weights["features.0.1.bias"],
                     weights["features.0.1.running_mean"], weights["features.0.1.running_var"])
    x = silu(x)

    # stages: features.1 .. features.7
    for stage_idx, (expand_ratio, kernel, stride, in_ch, out_ch, num_layers) in enumerate(STAGE_CONFIGS, start=1):
        for layer_i in range(num_layers):
            block_in = in_ch if layer_i == 0 else out_ch
            block_stride = stride if layer_i == 0 else 1
            prefix = f"features.{stage_idx}.{layer_i}"
            x = mbconv_block(x, weights, prefix, expand_ratio, kernel, block_stride, block_in, out_ch)

    # head: features.8
    x = conv2d(x, weights["features.8.0.weight"], None, stride=1, padding=0)
    x = batchnorm2d(x, weights["features.8.1.weight"], weights["features.8.1.bias"],
                     weights["features.8.1.running_mean"], weights["features.8.1.running_var"])
    x = silu(x)

    # AdaptiveAvgPool2d(1) + flatten
    x = x.mean(axis=(2, 3), keepdims=True)
    x = x.reshape(x.shape[0], -1)

    # classifier: Dropout is a no-op at inference, so only the Linear matters
    x = linear(x, weights["classifier.1.weight"], weights["classifier.1.bias"])
    return x


# ## Preprocessing
# Exact port of `CenterPad` / `get_padding` from `dataset.py`.


def get_padding(img_w, img_h):
    h_padding = (SIZE_W - img_w) / 2
    v_padding = (SIZE_H - img_h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    return int(l_pad), int(t_pad), int(r_pad), int(b_pad)


def center_pad(img, path=""):
    """
    Equivalent to transforms.Pad((l, t, r, b), fill=255, padding_mode='constant').
    Training images were always <= the target canvas, so padding was always
    non-negative. If a new image is larger than the canvas in some dimension,
    we downscale it (preserving aspect ratio) just enough to fit, then pad as
    usual -- and warn, since this is outside what the model was trained on.
    """
    l, t, r, b = get_padding(img.width, img.height)
    if min(l, t, r, b) < 0:
        scale = min(SIZE_W / img.width, SIZE_H / img.height)
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        print(f"WARNING: {path or 'image'} is {img.width}x{img.height}, larger than "
              f"the {SIZE_W}x{SIZE_H} canvas the model was trained on. Downscaling to "
              f"{new_size[0]}x{new_size[1]} before padding.")
        img = img.resize(new_size, Image.BILINEAR)
        l, t, r, b = get_padding(img.width, img.height)

    new_w = img.width + l + r
    new_h = img.height + t + b
    canvas = Image.new("RGB", (new_w, new_h), (PAD_FILL, PAD_FILL, PAD_FILL))
    canvas.paste(img, (l, t))
    return canvas


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = center_pad(img, path=path)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)
    return arr


# ## Prediction


def predict(path, weights):
    arr = load_image(path)
    batch = arr[np.newaxis, ...]
    logits = forward(batch, weights)
    probs = softmax(logits)
    pred = probs[0].argmax()
    pred_gender = CLASS_NAMES[pred]
    confidence = f"{probs[0, pred]*100:.1f}%"
    return pred_gender, confidence


def build_image_path(row):
    img_name = row.iloc[0]
    category = row.iloc[2]
    image_path = os.path.join(SD302_dict.get(category), img_name)
    true_gender = row.iloc[3]
    return image_path, true_gender
                              
def evaluate(df, weights):
    y_true = []
    y_pred = []
    for i, row in tqdm(df.iterrows(), total = df.shape[0], desc = "Processing: "):
        image_path, true_gender = build_image_path(row)
        pred_gender, _ = predict(image_path, weights)
        y_true.append(true_gender)
        y_pred.append(pred_gender)
        
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels = CLASS_NAMES, average = 'weighted')
    
    return y_true, y_pred, acc, f1

def make_confmat(y_true, y_pred, classes):
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis = 1)[:, None], index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot = True, cmap = "Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Predictions")
    plt.show()   
    
if __name__ == "__main__":
    weights = load_weights(WEIGHTS_PATH)
    df = pd.read_csv(test_csv_path)
    valid_categories = ("P", "R", "S")
    df = df[df.iloc[:, 2].isin(valid_categories)].reset_index(drop=True)
    y_true, y_pred, acc, f1 = evaluate(df, weights)
    print(f"\nAccuracy: {acc} | F1 score: {f1}")

make_confmat(y_true, y_pred, CLASS_NAMES)
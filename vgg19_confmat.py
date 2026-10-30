'''
For running VGG-19 on COMET on entire test dataset 
Displays confusion matrix, accuracy and F1 score
'''

import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sn


WEIGHTS_PATH = "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/vgg19_weights.npz"
test_csv_path = "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Documents/gender_test_set.csv"

SD302_dict = {
    "P": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/P/500/plain/png/",
    "R": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/R/500/slap-segmented/png/",
    "S": "C:/Users/htx-b7916ha/OneDrive - SG Govt M365/Downloads/S/500/slap-segmented/png/"
}

SIZE_W = 480   # target width  (matches SIZE_W in training)
SIZE_H = 640   # target height (matches SIZE_H in training)
PAD_FILL = 255  # white padding, matches transforms.Pad(..., fill=255)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = ["F", "M"]   # label 0 = F, label 1 = M, matches dataset.py

CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
       512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


def load_weights(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def conv2d(x, w, b, stride=1, padding=1):

    N, C, H, W = x.shape # input 
    C_out, _, kH, kW = w.shape # weights, k - kernal 

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                   mode="constant")

    H_p, W_p = x.shape[2], x.shape[3]
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    s0, s1, s2, s3 = x.strides
    shape = (N, C, out_h, out_w, kH, kW)
    strides = (s0, s1, s2 * stride, s3 * stride, s2, s3)
    patches = as_strided(x, shape=shape, strides=strides)
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, out_h * out_w, C * kH * kW)

    w_flat = w.reshape(C_out, C * kH * kW).T  # (C*kH*kW, C_out)

    out = patches @ w_flat  # (N, out_h*out_w, C_out)
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


def adaptive_avgpool2d(x, output_size):
    N, C, H, W = x.shape
    out_h, out_w = output_size
    out = np.zeros((N, C, out_h, out_w), dtype=x.dtype)
    h_starts = (np.arange(out_h) * H) // out_h
    h_ends = ((np.arange(out_h) + 1) * H + out_h - 1) // out_h
    w_starts = (np.arange(out_w) * W) // out_w
    w_ends = ((np.arange(out_w) + 1) * W + out_w - 1) // out_w
    for i in range(out_h):
        for j in range(out_w):
            region = x[:, :, h_starts[i]:h_ends[i], w_starts[j]:w_ends[j]]
            out[:, :, i, j] = region.mean(axis=(2, 3))
    return out


def linear(x, w, b):
    return x @ w.T + b


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, weights):

    idx = 0
    for v in CFG:
        if v == 'M':
            x = maxpool2d(x, 2, 2)
            idx += 1  # maxpool occupies one index
        else:
            w = weights[f"features.{idx}.weight"]
            b = weights[f"features.{idx}.bias"]
            x = conv2d(x, w, b, stride=1, padding=1)
            x = relu(x)
            idx += 2  # this conv + its following relu occupy two indices

    x = adaptive_avgpool2d(x, (5, 5))
    x = x.reshape(x.shape[0], -1)

    x = relu(linear(x, weights["classifier.0.weight"], weights["classifier.0.bias"]))
    x = relu(linear(x, weights["classifier.3.weight"], weights["classifier.3.bias"]))
    x = relu(linear(x, weights["classifier.6.weight"], weights["classifier.6.bias"]))
    x = linear(x, weights["classifier.9.weight"], weights["classifier.9.bias"])
    return x


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
        print(f"WARNING: {path or 'image'} is {img.width}x{img.height}, larger than "
              f"the {SIZE_W}x{SIZE_H} canvas the model was trained on. Downscaling to "
              f"{new_size[0]}x{new_size[1]} before padding. Predictions on resized "
              f"images may be less reliable than on images at original training scale.")
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
    arr = np.asarray(img, dtype=np.float32) / 255.0   # matches transforms.ToTensor()
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr


def predict(path, weights):
    arr = load_image(path)
    batch = arr[np.newaxis, ...] # adds batch dimension (3, H, W) -> (1, 3, H, W)
    logits = forward(batch, weights) # output: array([[x,x]]) - 2dim
    probs = softmax(logits)
    pred = probs[0].argmax() # returns index of max
    pred_gender = CLASS_NAMES[pred]
    confidence = f"{probs[0,pred]*100:.1f}%"
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
    
    
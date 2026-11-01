# **Project GRIP and AIFP — Web Deployment**

Two small Flask apps built for running on COMET laptop: 

1. Project GRIP: Fingerprint gender classification using pure NumPy implementation of the models  
2. Project AIFP: Normalisation of fingerprint images

## **Folder directory**

GRIP_AIFP_COMET/
├── README.md
├── fp_app.py
├── fp_app_norm.py
├── model_inference_effnetb3.py
├── model_inference_vgg19.py
├── effnetb3_confmat.py
├── effnetb3_npmodel.npz     # download separately (see Model weights)
├── vgg19_confmat.py
├── vgg19_weights.npz        # download separately (see Model weights)
├── templates/
│   ├── fp_preds.html
│   └── fp_norm.html
└── static/
    └── (icon images)

`fp_app.py` and `fp_app_norm.py` are independent apps that run on different ports — they're not meant to run as one process.

## **File descriptions**

| File | Description |
| ----- | ----- |
| `fp_app.py` | Flask app for fingerprint sex classification. Provides backend code and links it to interface (fp\_preds.html). Accepts an uploaded fingerprint image, runs it through whichever model backend is imported, and returns the predicted gender, confidence, and a Grad-CAM heatmap overlay. |
| `fp_app_norm.py` | Flask app for normalising fingerprint images. Provides backend code and links it to interface (fp\_app\_norm.html). Applies a variance-normalization contrast stretch to an uploaded image and returns before/after previews — no model involved, this is a standalone preprocessing preview tool. |
| `model_inference_effnetb3.py` | Pure-NumPy re-implementation of the EfficientNet-B3 backbone and classifier head, plus Grad-CAM, for running inference without PyTorch. Provides `run_inference` function to `fp_app.py`.  |
| `model_inference_vgg19.py` | Pure-NumPy re-implementation of the VGG-19 backbone and classifier head, plus Grad-CAM, for running inference without PyTorch. Provides `run_inference` function to `fp_app.py`.  |
| `effnetb3_npmodel.npz` | Trained EfficientNet-B3 weights, exported from PyTorch, converted from PyTorch tensors to NumPy arrays, for use by `model_inference_effnetb3.py`. |
| `vgg19_weights.npz` | Trained VGG-19 weights, exported from PyTorch, converted from PyTorch tensors to NumPy arrays, for use by `model_inference_vgg19.py`. |
| `effnetb3_confmat.py` | For testing imported EfficientNetB3 weights on COMET using test dataset. Displays confusion matrix, accuracy and F1 score. |
| `vgg19_confmat.py` | For testing imported VGG19 weights on COMET using test dataset.  Displays confusion matrix, accuracy and F1 score. |
| `templates/fp_preds.html` | Frontend page for `fp_app.py` — upload image and results display. |
| `templates/fp_norm.html` | Frontend page for `fp_app_norm.py` — upload image and before/after preview. |
| `static/` | Icon images used by the HTML templates. |

## **Requirements**

pip install flask numpy pillow matplotlib scipy

`scipy` is only needed by `model_inference_effnetb3.py` (for `expit`, used in SiLU/sigmoid). `model_inference_vgg19.py` doesn't need it.

## **Model weights**

The trained weights are too large for GitHub, so they are hosted on Google Drive:

| File | Model | Download |
| ----- | ----- | ----- |
| `vgg19_weights.npz` | VGG-19 | [[Google Drive link](PASTE_LINK_HERE)] https://drive.google.com/file/d/1xbys2nqieZaoKYOaRczM0zPbD4wBrNlC/view?usp=share_link |
| `effnetb3_npmodel.npz` | EfficientNet-B3 | [[Google Drive link](PASTE_LINK_HERE)](https://drive.google.com/file/d/15ECYzbZMNYegqRTVtudNe8VBICQPUWS0/view?usp=sharing) |

Download both files and place them in the project root, next to `fp_app.py` as shown under **Folder directory**

## **Usage**

1. Download the weights (see **Model weights** above) and transfer them to the COMET laptop.
   Each backend looks for its `.npz` weights file relative to the process's working directory:

| Backend | Default filename | Env var override |
| ----- | ----- | ----- |
| VGG-19 | `vgg19_weights.npz` | `FP_MODEL_WEIGHTS_VGG19` |
| EfficientNet-B3 | `effnetb3_npmodel2npz` | *(none — see below)* |

2. Edit `WEIGHTS_PATH, test_csv_path, SD302_dict` in effnetb3\_confmat.py or vgg19\_confmat.py. Run script to test model.   
3. Edit `WEIGHTS_PATH` in `model_inference_vgg19.py` or `model_inference_effnetb3.py`

4. Choose model backend  
   `fp_app.py` currently imports `model_inference_vgg19`:  
   import model\_inference\_vgg19  
   ...  
   result \= model\_inference\_vgg19.run\_inference(pil\_image)  
     
   To switch to the EfficientNet-B3 backend, change both lines to `model_inference_effnetb3`. Both modules expose the same interface:  
   run\_inference(pil\_image, alpha=0.45) \-\> {  
       "prediction": "F" | "M",  
       "confidence": float,   \# 0-100  
       "input\_png": bytes,  
       "heatmap\_png": bytes,  \# Grad-CAM overlay  
   }

   so no other changes to `fp_app.py` are needed either way.

5. Run fp\_app.py or fp\_app\_norm.py in Spyder.   
   **Prediction app** (classifies an uploaded fingerprint and returns a Grad-CAM heatmap):  
   python fp\_app.py  
     
   Serves at `http://127.0.0.1:5000`  
   **Normalization app** (preview-only variance-normalization contrast stretch, no model involved):  
   python fp\_app\_norm.py  
     
   Serves at `http://127.0.0.1:5001`  
   Copy and paste the corresponding link from the terminal into browser to run app.

6. To quit app, enter CTRL+C in terminal

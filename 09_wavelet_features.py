import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets library
from ultralytics import YOLO
import os, random

def extract_wavelet_features(img_folder):
    # 1. Setup Model and Image
    model = YOLO('yolov8s.pt')
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    
    # 2. Get YOLO Detections
    results = model.predict(img_path, classes=[2, 7], conf=0.3)
    
    if len(results[0].boxes) == 0:
        print("No vehicles detected to decompose.")
        return

    # 3. Take the first detected vehicle and crop it (the "Patch")
    box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    vehicle_patch = img_bgr[y1:y2, x1:x2]
    gray_patch = cv2.cvtColor(vehicle_patch, cv2.COLOR_BGR2GRAY)
    
    # 4. 2D Discrete Wavelet Transform (DWT)
    # 'haar' is a simple, effective wavelet for edge/texture decomposition
    coeffs2 = pywt.dwt2(gray_patch, 'haar')
    LL, (LH, HL, HH) = coeffs2 # LL: Approximation, LH/HL/HH: Details

    # 5. Create a Feature Vector
    # We flatten the Detail coefficients to create a unique 'fingerprint'
    feature_vector = np.hstack([LH.flatten(), HL.flatten(), HH.flatten()])
    print(f"Vehicle Feature Vector Size: {feature_vector.shape[0]} elements")

    # --- Visualization ---
    titles = ['Original Patch', 'Approximation (LL)', 'Horizontal Detail (LH)', 
              'Vertical Detail (HL)', 'Diagonal Detail (HH)']
    images = [gray_patch, LL, LH, HL, HH]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.suptitle("Step 9: Image Decomposition for Feature Fingerprinting", fontsize=16)
    plt.show()

# --- RUN ---
extract_wavelet_features("data/kitti_lite")
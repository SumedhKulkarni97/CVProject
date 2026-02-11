import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt  # pip install PyWavelets
from ultralytics import YOLO  # pip install ultralytics
import os, random

"""
PROJECT: EagleEye ADAS - Step 10: Object Classification Refinement
GOAL: Use high-dimensional Wavelet features to distinguish between specific vehicle types.

CONCEPTS:
- Feature Vector: The flattened wavelet coefficients from Step 9.
- Refined Classification: Moving from general 'vehicle' to specific 'Truck' or 'Van'.
"""


def adas_classification_pipeline(img_folder):
    # 1. Setup - Use 'Small' model for better precision than 'Nano'
    model = YOLO('yolov8s.pt') 
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. YOLO Detection with lower confidence to catch harder-to-see cars
    results = model.predict(img_path, classes=[2, 7], conf=0.3)
    detections = results[0].boxes.xyxy.cpu().numpy()
    
    if len(detections) == 0:
        print("No vehicles detected to classify.")
        return

    # 3. Setup Visualization Grid
    num_objs = len(detections)
    fig, axes = plt.subplots(1, num_objs, figsize=(4 * num_objs, 4), squeeze=False)

    # 4. Iterate through each detection
    for i, box in enumerate(detections):
        x1, y1, x2, y2 = box.astype(int)
        
        # A. Extract Patch
        patch_gray = gray[y1:y2, x1:x2]
        patch_rgb = img_rgb[y1:y2, x1:x2]
        
        # B. Wavelet Decomposition (Haar)
        # We use the 'Diagonal Detail' (HH) to measure texture complexity
        coeffs2 = pywt.dwt2(patch_gray, 'haar')
        LL, (LH, HL, HH) = coeffs2
        
        # C. Refined Logic: Trucks/Buses typically have higher frequency 
        # detail due to larger size and complex structural textures.
        texture_score = np.mean(np.abs(HH)) 
        
        # Threshold logic for refinement
        if texture_score > 5.0 or (x2 - x1) > 200:
            refined_label = "Truck/Bus"
            color = (255, 0, 0) # Red for large
        else:
            refined_label = "Passenger Car"
            color = (0, 255, 0) # Green for small

        # D. Matplotlib Display
        axes[0, i].imshow(patch_rgb)
        axes[0, i].set_title(f"Refined: {refined_label}\nScore: {texture_score:.2f}", fontsize=10)
        axes[0, i].axis('off')

    plt.suptitle(f"Step 10: Wavelet-Based Classification Verification\nSample: {os.path.basename(img_path)}", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite"
adas_classification_pipeline(IMG_DIR)
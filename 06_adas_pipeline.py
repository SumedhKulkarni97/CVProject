import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, random

"""
PROJECT: EagleEye ADAS - Step 6: Integrated Pipeline Dashboard
GOAL: Combine all previous knowledge into a single processing flow.
"""

def run_adas_pipeline(img_folder):
    # 1. Selection
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    
    # --- PROCESSING STEPS ---
    # Step A: Preprocessing (Grayscale + Bilateral Filter)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Step B: Segmentation (Road ROI)
    # Creating a simple trapezoidal ROI mask often used in lane keeping
    mask = np.zeros_like(gray)
    polygon = np.array([[(0, h), (w, h), (int(w*0.6), int(h*0.4)), (int(w*0.4), int(h*0.4))]])
    cv2.fillPoly(mask, polygon, 255)
    
    # Step C: Edge Detection (Canny)
    edges = cv2.Canny(filtered, 50, 150)
    road_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Step D: Corner Detection (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(filtered, 100, 0.08, 25, mask=mask)
    
    # --- VISUALIZATION ---
    img_display = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if corners is not None:
        for i in np.intp(corners): # Using np.intp to avoid your previous error
            x, y = i.ravel()
            cv2.circle(img_display, (x, y), 7, (0, 0, 255), -1)

    # Creating the 4-panel Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Raw Sensor Input")
    
    axes[0, 1].imshow(filtered, cmap='gray')
    axes[0, 1].set_title("2. Filtered Intensity (Bilateral)")
    
    axes[1, 0].imshow(road_edges, cmap='gray')
    axes[1, 0].set_title("3. Extracted Lane Edges (ROI)")
    
    axes[1, 1].imshow(img_display)
    axes[1, 1].set_title("4. Integrated Feature Map (Corners)")

    for ax in axes.flat: ax.axis('off')
    plt.suptitle(f"EagleEye ADAS Pipeline | Sample: {os.path.basename(img_path)}", fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- RUN ---
run_adas_pipeline("data/kitti_lite")
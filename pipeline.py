import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from ultralytics import YOLO

def plot_adas_grid(img_folder):
    # 1. Select a random image from KITTI Lite
    files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))]
    img_name = random.choice(files)
    img_path = os.path.join(img_folder, img_name)
    
    # --- PROCESSING STEPS ---
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Step 02: Noise Reduction
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    blurred_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    
    # Step 03: Road Segmentation
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    road_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 50, 200]))
    
    # Step 04: Canny Edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Step 05: ROI Masking
    h, w = gray.shape
    roi_mask = np.zeros_like(edges)
    cv2.fillPoly(roi_mask, [np.array([[(0, h), (w, h), (w, int(h*0.45)), (0, int(h*0.45))]])], 255)
    masked_edges = cv2.bitwise_and(edges, roi_mask)
    
    # Step 06: Lane Refinement
    refined = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # Step 07: Shi-Tomasi Corners
    corners = cv2.goodFeaturesToTrack(gray, mask=roi_mask, maxCorners=50, qualityLevel=0.01, minDistance=10)
    corner_img = img_rgb.copy()
    if corners is not None:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(corner_img, (int(x), int(y)), 8, (255, 0, 0), -1)
            
    # Step 08: YOLO Detection
    model = YOLO('yolov8s.pt')
    results = model.predict(img_path, classes=[2, 7], conf=0.3, verbose=False)
    yolo_img = img_rgb.copy()
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # --- PLOTTING 2 ROWS OF 4 IMAGES ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Data mapping for the grid
    step_data = [
        (img_rgb, "01: Original", None), (blurred_rgb, "02: Blur", None),
        (road_mask, "03: Road Seg", "gray"), (edges, "04: Canny", "gray"),
        (masked_edges, "05: ROI Mask", "gray"), (refined, "06: Lane Refine", "gray"),
        (corner_img, "07: Features", None), (yolo_img, "08: YOLO Detect", None)
    ]

    for i, (img, title, cmap) in enumerate(step_data):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(img, cmap=cmap)
        axes[row, col].set_title(title, fontsize=16, fontweight='bold')
        axes[row, col].axis('off')

    plt.suptitle(f"ADAS Perception Grid: {img_name}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run on KITTI Lite
plot_adas_grid("data/kitti_lite")
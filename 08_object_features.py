import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, random
from ultralytics import YOLO

"""
PROJECT: EagleEye ADAS - Step 8: Localized Feature Extraction
GOAL: Find stable Shi-Tomasi corners only inside detected vehicle boxes.

CONCEPTS:
- Sub-pixel Accuracy: Finding features within a specific object mask.
- Feature Locking: These corners will stay 'attached' to the car as it moves.
"""

def extract_object_features(img_folder):
    # 1. Load Model and Image
    model = YOLO('yolov8s.pt') 
    """
    To resolve the detection errors observed, I upgraded from YOLOv8n to YOLOv8s. 
        (YOLOv8n - nano, high speed, low resource usage, lightweight. YOLOv8s - small, balanced speed, higher precision, more complex.)
    This increased the model's parameter count, allowing for better feature extraction in complex environments. 
    Additionally, tuning the confidence threshold to 0.3 ensured that vehicles with lower contrast were successfully captured while maintaining real-time performance.
    """
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Get YOLO Detections
    results = model.predict(img_path, classes=[2, 7], conf=0.3) # Cars and Trucks only
    
    # 3. Create a mask specifically for detected objects
    obj_mask = np.zeros_like(gray)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() # [xmin, ymin, xmax, ymax]
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Draw a white rectangle on the mask for each car
            cv2.rectangle(obj_mask, (x1, y1), (x2, y2), 255, -1)
            # Draw a green bounding box on display image
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 4. Shi-Tomasi inside the Object Mask
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, 
                                      qualityLevel=0.05, minDistance=10, 
                                      mask=obj_mask)

    # 5. Draw the 'Anchor Points'
    if corners is not None:
        for i in np.intp(corners):
            x, y = i.ravel()
            cv2.circle(img_rgb, (x, y), 5, (0, 0, 255), -1) # Blue Dots

    # --- Visualization ---
    plt.figure(figsize=(15, 8))
    plt.imshow(img_rgb)
    plt.title("Shi-Tomasi Corners Localized within YOLO Bounding Boxes")
    plt.axis('off')
    plt.show()

# --- RUN ---
extract_object_features("data/kitti_lite")
import cv2
import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO

"""
PROJECT: EagleEye ADAS - Step 7: Object Detection (YOLO)
GOAL: Detect and label objects (cars, pedestrians) using a Deep Learning model.

CONCEPTS:
- Bounding Boxes: Coordinates [x, y, w, h] that enclose a detected object.
- Confidence Score: The model's 'certainty' that an object belongs to a class.
- YOLO: A Convolutional Neural Network (CNN) optimized for real-time speed.
"""

def detect_objects_yolo(img_folder):
    # 1. Initialize YOLOv8 (using the 'nano' version for speed)
    model = YOLO('yolov8n.pt') 

    # 2. Select a Random KITTI Image
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    
    # 3. Run Inference
    # We pass the original image to get the most accurate detections
    results = model(img_path)

    # 4. Process Results
    # YOLO's plot() method draws bounding boxes and labels for us
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # 5. Extract specific data for future logic
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} objects in {os.path.basename(img_path)}")

    # Visualization
    plt.figure(figsize=(15, 8))
    plt.imshow(annotated_img_rgb)
    plt.title(f"Deep Learning - Object Detection (YOLOv8)", fontsize=16)
    plt.axis('off')
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite"
detect_objects_yolo(IMG_DIR)

"""
step-by-step breakdown of the YOLO logic:
1. Grid Division The algorithm divides the input image into an SxS grid.
Each grid cell is responsible for detecting an object if the center of that object falls within the cell.
This allows the model to process the entire spatial context of the image at once.

2. Bounding Box Prediction (x, y, w, h)
For each grid cell, the network predicts bounding boxes and a confidence score for those boxes.
x, y: Represent the center of the box relative to the grid cell.
w, h: Represent the width and height of the box relative to the whole image.
Confidence Score: Reflects how likely the box contains an object and how accurate the box is.

3. Class Probability Map Simultaneously, each grid cell predicts conditional class probabilities (e.g., is this a car, a pedestrian, or a truck?).
YOLOv8 uses a specialized "Head" to separate the tasks of finding the box (localization) and naming the object (classification), which improves accuracy.

4. Non-Maximum Suppression (NMS)Because multiple grid cells might predict the same object, the raw output often contains overlapping "candidate" boxes.
NMS is a post-processing step that cleans this up.It looks at all overlapping boxes, keeps the one with the highest confidence score, 
and removes the others that overlap significantly (based on Intersection over Union or IoU).
"""
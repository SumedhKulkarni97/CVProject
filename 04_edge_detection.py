import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

"""
PROJECT: EagleEye ADAS - Step 4: Lane Line Extraction
GOAL: Isolate lane markings by applying Canny edges to a Segmented ROI.

CONCEPTS:
- ROI (Region of Interest): Masking the image to process only the road.
- Canny Edge Detection: A multi-stage algorithm to detect a wide range of edges.
- Bitwise Logic: Using 'AND' operations to combine masks and images.
"""

def detect_lanes(img_folder):
    # 1. Load Random KITTI Sample
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Create the Road ROI (using the logic from Step 3)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 50, 180])
    road_mask = cv2.inRange(img_hsv, lower_road, upper_road)
    
    # 3. Apply Canny Edge Detection
    # We use a lower threshold of 50 and upper of 150 for KITTI asphalt
    edges = cv2.Canny(gray, 50, 150)

    # 4. Combine: Apply the ROI to the Edges
    # This removes all edges (trees, buildings) that aren't on the road
    lane_roi_edges = cv2.bitwise_and(edges, edges, mask=road_mask)

    # --- Visualization ---
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Scene")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(road_mask, cmap='gray')
    plt.title("2. Road Segmentation (ROI)")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("3. Global Canny Edges (Noisy)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(lane_roi_edges, cmap='gray')
    plt.title("4. Refined Lane Edges (ROI Applied)")
    plt.axis('off')

    plt.suptitle("ROI-Based Edge Detection for Lane Finding", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite"
detect_lanes(IMG_DIR)


"""
How Canny Edge Detection works:
1. Noise Reduction (Gaussian Blur)
Since mathematical derivatives are highly sensitive to pixel noise, the first step is to "smooth" the image.
A 5x5 Gaussian filter is typically applied to the grayscale image.
This prevents random speckles or sensor grain from being incorrectly identified as "edges".

2. Finding Intensity Gradients (Sobel Kernel)
The algorithm calculates how quickly the pixel values change in both the horizontal (G_x) and vertical (G_y) directions.
It uses a Sobel Kernel to find the edges' magnitude (strength) and direction (angle).Magnitude ($G$): G = \sqrt{G_x^2 + G_y^2}
Direction (theta): \arctan({G_y}/{G_x})

3. Non-Maximum Suppression (Thinning)
After finding the gradients, the edges often look "blurry" or "thick".
The algorithm scans the image and checks if a pixel has the highest gradient value compared to its neighbors in the direction of the gradient.
If it isn't the local maximum, it is set to zero (suppressed). This results in thin, 1-pixel wide "skeletons".

4. Hysteresis Thresholding (The Final Filter)
This is the most critical stage where we use the two values you set in your code (e.g., 50 and 150):
Strong Edges: Pixels with a gradient higher than the upper threshold (150) are immediately accepted as edges.
Discarded Pixels: Pixels below the lower threshold (50) are immediately rejected.
Weak Edges: Pixels between the two values are only kept if they are connected to a "Strong Edge".
"""
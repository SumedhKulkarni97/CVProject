import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

"""
PROJECT: EagleEye ADAS - Step 3: Semantic Segmentation Study
GOAL: Isolate the drivable road area and automobiles using color-space masks.

CONCEPTS:
- HSV Color Space: Better than RGB for segmentation; Hue (H) is stable under shadows.
- Morphological Closing: A non-linear operation that 'heals' gaps in segmented regions.
- Otsu's Thresholding: Automatically finds the best intensity cutoff for segmentation.
"""

def perform_segmentation(img_folder):
    # 1. Random Image Selection for variety
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Road Segmentation via HSV Thresholding
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Defining typical gray asphalt ranges (Hue, Saturation, Value)
    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 50, 180])
    road_mask = cv2.inRange(img_hsv, lower_road, upper_road)

    # 3. Refine Segmentation with Morphology
    kernel = np.ones((7, 7), np.uint8)
    # Closing (Dilation then Erosion) fills small holes in the road mask
    refined_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)

    # 4. Otsu's Segmentation for Objects
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Visualization ---
    titles = ['Original RGB', 'Road Mask (HSV)', 'Refined (Morphology)', "Otsu's (Objects)"]
    images = [img_rgb, road_mask, refined_mask, otsu_mask]

    plt.figure(figsize=(20, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    
    plt.suptitle("Classical Image Segmentation Analysis", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite" 
perform_segmentation(IMG_DIR)

"""
Road Mask (HSV): You learn that color is relative. By looking at this mask, you see that the computer has isolated the "grayness" of the asphalt.
We use the HSV (Hue, Saturation, Value) color space because it is more robust to shadows than RGB. Even if a tree casts a shadow on the road, the "Hue" (the essence of the gray) remains similar, allowing the ADAS system to keep "seeing" the road.

Refined Mask (Morphology): You learn that raw data is "noisy." The initial mask often has holes (black spots) where there might be a manhole cover or a different colored patch of road.
We use Morphological Closing (a non-linear operator) to "heal" these holes. This ensures the car's path-planning algorithm sees a solid surface to drive on rather than a fragmented one.

Otsu’s Thresholding (Objects): You learn how the computer automatically finds the "separation point" in a scene.
Otsu’s method calculates the optimal intensity threshold without you having to guess a number. It is excellent for "cutting out" dark objects (like cars or tires) from a bright background (like the sky or road).
"""
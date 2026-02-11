import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt

"""
PROJECT: EagleEye ADAS - Step 2: Random Filter Evaluation
GOAL: Gain intuition on how filters affect object boundaries across different scenes.
"""

def filters(img_folder):
    # 1. Randomly pick an image from 1000-pack
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    if not all_imgs:
        print("No images found in the directory")
        return
    
    random_img_name = random.choice(all_imgs)
    img_path = os.path.join(img_folder, random_img_name)
    
    # Load in grayscale
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Apply Filters
    # Linear: Good for general noise, but blurs edges.
    gaussian = cv2.GaussianBlur(original, (7, 7), 0)
    
    # Non-Linear: Median is the 'King' of removing impulse noise.
    median = cv2.medianBlur(original, 7)
    
    # Non-Linear: Bilateral preserves edges while smoothing flat areas.
    bilateral = cv2.bilateralFilter(original, 9, 75, 75)

    # 3. Side-by-Side Visualization
    titles = [f'Original ({random_img_name})', 'Gaussian (Linear)', 
              'Median (Non-Linear)', 'Bilateral (Edge-Preserving)']
    images = [original, gaussian, median, bilateral]

    plt.figure(figsize=(20, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        # We use a subset/zoom to see the filter effect clearly on a car or lane
        # Zooming into the middle of the KITTI frame (where cars usually are)
        h, w = original.shape
        zoom_y, zoom_x = int(h*0.4), int(w*0.3)
        plt.imshow(images[i][zoom_y:zoom_y+200, zoom_x:zoom_x+400], cmap='gray')
        plt.title(titles[i], fontsize=14)
        plt.axis('off')
    
    plt.suptitle("Filter Effects on Object Boundaries", fontsize=18)
    # We use top=0.95 to reserve 5% of the window height for the plt.suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite" 
filters(IMG_DIR)
import os
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

"""
PROJECT: EagleEye ADAS - Step 1: Exploratory Data Analysis (EDA)
DATASET: KITTI Vision Benchmark Suite (Object Detection Subset)

KITTI DETAILS FOR DOCUMENTATION:
- Images are rectified and synchronized 
- Coordinate System: x (right), y (down), z (forward)
- 2D Bounding Box format: [left, top, right, bottom] (pixel coords)
"""

def get_kitti_stats(lbl_folder):
    """Parses all label files to generate statistics."""
    class_counts = Counter()
    total_files = 0
    
    label_files = [f for f in os.listdir(lbl_folder) if f.endswith('.txt')]
    for lbl_file in label_files:
        total_files += 1
        with open(os.path.join(lbl_folder, lbl_file), 'r') as f:
            for line in f:
                obj_type = line.split()[0]
                if obj_type != "DontCare":
                    class_counts[obj_type] += 1
    return class_counts, total_files

def visualize_sample(img_folder, lbl_folder, sample_id=None):
    """Visualizes a single image with its ground-truth labels."""
    if sample_id is None:
        all_imgs = [f.split('.')[0] for f in os.listdir(img_folder) if f.endswith('.png')]
        sample_id = random.choice(all_imgs)

    img_path = os.path.join(img_folder, f"{sample_id}.png")
    lbl_path = os.path.join(lbl_folder, f"{sample_id}.txt")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

    # --- Subplot 1: Image Visualization ---
    ax1.imshow(img)
    ax1.set_title(f"KITTI Sample: {sample_id}.png")
    
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                data = line.split()
                obj_type = data[0]
                if obj_type == "DontCare": continue
                
                # [left, top, right, bottom]
                l, t, r, b = map(float, data[4:8])
                rect = patches.Rectangle((l, t), r-l, b-t, linewidth=2, edgecolor='lime', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(l, t-5, obj_type, color='white', weight='bold', bbox=dict(facecolor='lime', alpha=0.5))
    ax1.axis('off')

    # --- Subplot 2: Dataset Wide Distribution ---
    stats, count = get_kitti_stats(lbl_folder)
    ax2.bar(stats.keys(), stats.values(), color='teal')
    ax2.set_title(f"Dataset Distribution ({count} files analyzed)")
    ax2.set_ylabel("Frequency")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
# Ensure these paths match your VS Code setup
IMG_DIR = "data/kitti_lite"
LBL_DIR = "data/kitti_labels"

if __name__ == "__main__":
    visualize_sample(IMG_DIR, LBL_DIR)
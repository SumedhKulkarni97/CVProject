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
- Images are rectified/synchronized (Left Color Image_2).
- Coordinates: [left, top, right, bottom] in pixels.
- Metadata: Includes occlusion (0-3) and truncation (0-1.0).
"""

def get_detailed_stats(img_folder, lbl_folder):
    """Generates a full report on data usage and label variety."""
    class_counts = Counter()
    total_images = len([f for f in os.listdir(img_folder) if f.endswith('.png')])
    total_labels = len([f for f in os.listdir(lbl_folder) if f.endswith('.txt')])
    
    # Calculate disk usage in Megabytes
    img_size_mb = sum(os.path.getsize(os.path.join(img_folder, f)) for f in os.listdir(img_folder)) / (1024*1024)
    
    for lbl_file in os.listdir(lbl_folder):
        if lbl_file.endswith('.txt'):
            with open(os.path.join(lbl_folder, lbl_file), 'r') as f:
                for line in f:
                    obj_type = line.split()[0]
                    if obj_type != "DontCare":
                        class_counts[obj_type] += 1
                        
    return {
        "img_count": total_images,
        "lbl_count": total_labels,
        "disk_mb": round(img_size_mb, 2),
        "classes": class_counts,
        "unique_labels": len(class_counts)
    }

def visualize_with_report(img_folder, lbl_folder):
    stats = get_detailed_stats(img_folder, lbl_folder)
    
    # --- Print Documentation Report ---
    print("-" * 30)
    print("KITTI DATASET SUMMARY REPORT")
    print("-" * 30)
    print(f"Total Images Used:    {stats['img_count']}")
    print(f"Total Labels Used:    {stats['lbl_count']}")
    print(f"Total Storage Size:   {stats['disk_mb']} MB")
    print(f"Unique Object Types:  {stats['unique_labels']}")
    print("\nLabel Breakdown:")
    for obj, count in stats['classes'].items():
        print(f"- {obj:12}: {count} instances")
    print("-" * 30)

    # --- Visualization ---
    sample_id = random.choice([f.split('.')[0] for f in os.listdir(img_folder)])
    img = cv2.cvtColor(cv2.imread(os.path.join(img_folder, f"{sample_id}.png")), cv2.COLOR_BGR2RGB)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.imshow(img)
    ax1.set_title(f"Visualizing Sample: {sample_id}.png")
    
    lbl_path = os.path.join(lbl_folder, f"{sample_id}.txt")
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                data = line.split()
                if data[0] == "DontCare": continue
                l, t, r, b = map(float, data[4:8])
                rect = patches.Rectangle((l, t), r-l, b-t, linewidth=2, edgecolor='lime', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(l, t-5, data[0], color='white', weight='bold', bbox=dict(facecolor='lime', alpha=0.5))
    ax1.axis('off')

    ax2.bar(stats['classes'].keys(), stats['classes'].values(), color='darkorange')
    ax2.set_title("Label Frequency Across Dataset")
    plt.xticks(rotation=45)
    plt.show()

# --- RUN ---
IMG_DIR = "data/kitti_lite"
LBL_DIR = "data/kitti_labels"

if __name__ == "__main__":
    visualize_with_report(IMG_DIR, LBL_DIR)
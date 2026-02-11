# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# """
# PROJECT: EagleEye ADAS - Step 11: Final Temporal Tracking
# GOAL: Calculate motion vectors on real sequential KITTI frames.
# """

# def adas_temporal_tracking(seq_path):
#     # 1. Access the specific 'data' folder from your screenshot
#     all_imgs = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
    
#     if len(all_imgs) < 2:
#         print("Not enough consecutive frames found!")
#         return

#     # Load two strictly consecutive frames
#     img1_bgr = cv2.imread(os.path.join(seq_path, all_imgs[0]))
#     img2_bgr = cv2.imread(os.path.join(seq_path, all_imgs[1]))

#     # Preprocessing
#     img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
#     img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
#     gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

#     # 2. Identify trackable features (Focus on road/vehicles)
#     h, w = gray1.shape
#     mask = np.zeros_like(gray1)
#     mask[int(h*0.4):h, :] = 255 
#     p0 = cv2.goodFeaturesToTrack(gray1, mask=mask, maxCorners=150, 
#                                  qualityLevel=0.01, minDistance=10)

#     # 3. Lucas-Kanade Parameters
#     lk_params = dict(winSize=(21, 21), maxLevel=3,
#                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

#     # 4. Calculate Flow
#     p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

#     # 5. Visualization Setup
#     res_img = img2_rgb.copy()
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]

#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel().astype(int)
#         c, d = old.ravel().astype(int)
#         # Green line = direction and speed of motion
#         cv2.line(res_img, (a, b), (c, d), (0, 255, 0), 2)
#         # Red dot = new position of the object
#         cv2.circle(res_img, (a, b), 3, (255, 0, 0), -1)

#     # 6. Display 3-Panel Verification
#     fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
#     axes[0].imshow(img1_rgb)
#     axes[0].set_title(f"Frame T: {all_imgs[0]}")
#     axes[0].axis('off')

#     axes[1].imshow(img2_rgb)
#     axes[1].set_title(f"Frame T+1: {all_imgs[1]}")
#     axes[1].axis('off')

#     axes[2].imshow(res_img)
#     axes[2].set_title("Result: Sparse Optical Flow")
#     axes[2].axis('off')

#     plt.suptitle("EagleEye ADAS: Temporal Motion Analysis", fontsize=16)
#     plt.tight_layout()
#     plt.show()

# # --- RUN ---
# # Updated path based on your uploaded folder structure
# SEQUENCE_DATA_PATH = "data/2011_09_26/2011_09_26_drive_0020_sync/image_02/data"
# adas_temporal_tracking(SEQUENCE_DATA_PATH)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

"""
PROJECT: EagleEye ADAS - Step 11: Real-World Velocity Estimation
GOAL: Convert pixel displacement into km/h using KITTI timestamps.
"""

def get_time_diff(timestamps_path, idx1, idx2):
    """Calculates the time difference in seconds between two KITTI frames."""
    with open(timestamps_path, 'r') as f:
        lines = f.readlines()
        # KITTI timestamps are in format: YYYY-MM-DD HH:MM:SS.fffffffff
        t1 = datetime.strptime(lines[idx1].strip()[:-3], '%Y-%m-%d %H:%M:%S.%f')
        t2 = datetime.strptime(lines[idx2].strip()[:-3], '%Y-%m-%d %H:%M:%S.%f')
        return (t2 - t1).total_seconds()

def calculate_velocity_adas(seq_path, timestamps_path):
    all_imgs = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
    
    # 1. Get Delta Time (dt)
    dt = get_time_diff(timestamps_path, 0, 1)
    
    img1 = cv2.imread(os.path.join(seq_path, all_imgs[0]))
    img2 = cv2.imread(os.path.join(seq_path, all_imgs[1]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 2. Track Points
    p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=50, qualityLevel=0.01, minDistance=10)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # 3. Calculate Speed (Relative)
    # In a real system, we'd use camera calibration to convert pixels to meters.
    # Here, we use a constant 'pixel_to_meter' scale factor for KITTI (approx 0.05m/px)
    px_to_m = 0.05 
    
    res_img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    for i, (new, old) in enumerate(zip(p1[st==1], p0[st==1])):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        
        # Euclidean distance in pixels
        dist_px = np.sqrt((a-c)**2 + (b-d)**2)
        
        # Velocity = (Distance * Scale) / Time
        v_mps = (dist_px * px_to_m) / dt
        v_kmh = v_mps * 3.6 # Convert m/s to km/h
        
        # Draw vectors and speed text
        cv2.line(res_img, (a, b), (c, d), (0, 255, 0), 2)
        if v_kmh > 5: # Only label moving objects
            cv2.putText(res_img, f"{int(v_kmh)}km/h", (a, b-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    plt.figure(figsize=(12, 6))
    plt.imshow(res_img)
    plt.title(f"Velocity Estimation (dt = {dt:.3f}s)")
    plt.axis('off')
    plt.show()

# --- PATHS FROM YOUR FOLDER STRUCTURE ---
BASE_DIR = "data/2011_09_26/2011_09_26_drive_0020_sync"
DATA_PATH = os.path.join(BASE_DIR, "image_02/data")
TIME_PATH = os.path.join(BASE_DIR, "image_02/timestamps.txt")

calculate_velocity_adas(DATA_PATH, TIME_PATH)
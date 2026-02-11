import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, random

def detect_corners_fixed(img_folder):
    # 1. Selection & Loading
    all_imgs = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    img_path = os.path.join(img_folder, random.choice(all_imgs))
    img_bgr = cv2.imread(img_path)
    h, w = img_bgr.shape[:2]
    
    # 2. Preprocessing
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Smooths noise

    # 3. Create an ROI Mask (Ignore the Sky)
    # This prevents corners from being detected in the bright sky or image edges
    mask = np.zeros_like(gray)
    mask[int(h*0.4):h, :] = 255 

    # 4. Apply Shi-Tomasi
    """
    OpenCV has a function, cv.goodFeaturesToTrack(). It finds N strongest corners in the image by Shi-Tomasi method (or Harris Corner Detection, if you specify it). 
    As usual, image should be a grayscale image. Then you specify number of corners you want to find. 
    Then you specify the quality level, which is a value between 0-1, which denotes the minimum quality of corner below which everyone is rejected.
    """
    corners = cv2.goodFeaturesToTrack(
        blurred, 
        maxCorners=80, 
        qualityLevel=0.08, # High quality to avoid weak noise
        minDistance=25, 
        mask=mask # Only look at the road/cars
    )

    # 5. Drawing & Visualization
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    if corners is not None:
        corners = np.intp(corners) 
        
        for i in corners:
            x, y = i.ravel()
            # Draw Blue dots (0, 0, 255) for tracking features
            cv2.circle(img_rgb, (x, y), 6, (0, 0, 255), -1) 

    plt.figure(figsize=(16, 8))
    plt.imshow(img_rgb)
    plt.title("Shi-Tomasi Corners (Blue Dots) - Sky Noise Ignored", fontsize=15)
    plt.axis('off')
    plt.show()

# --- RUN ---
detect_corners_fixed("data/kitti_lite")

"""
How does the Shi-Tomasi Corner Detector works: 
1. The "Window" Concept
The algorithm slides a small window (e.g., 3x3 or 5x5 pixels) across your KITTI image. It asks: "If I move this window slightly in any direction, does the content inside change significantly?"
    Flat Region: Moving the window in any direction results in no change.
    Edge: Moving the window along the edge results in no change; only moving across the edge shows change.
    Corner: Moving the window in any direction results in a massive change in intensity.

2. The Math: The Second Moment Matrix (M)
To quantify this "change," the algorithm calculates the intensity gradients ($I_x$ and $I_y$) for the pixels inside the window. 
It then builds a matrix M. This matrix describes how the image intensity changes in the x and y directions.

3. Finding Eigenvalues (lambda_1, lambda_2)
The algorithm calculates the eigenvalues (lambda_1 and lambda_2) of this matrix M. 
Think of these eigenvalues as the "strength" of the intensity change in two perpendicular directions:
    If both lambda_1 and lambda_2 are small: It's a flat region.
    If one is large and one is small: It's an edge.
    If both are large: It's a corner.
    
4. The Shi-Tomasi Innovation (The Scoring Function)
While the Harris detector uses a complex combination of these values, Shi and Tomasi discovered that you only need to look at the smaller of the two eigenvalues to find the most stable corners.
Their scoring formula ($R$) is simple: R = min(lambda_1, lambda_2)
"""
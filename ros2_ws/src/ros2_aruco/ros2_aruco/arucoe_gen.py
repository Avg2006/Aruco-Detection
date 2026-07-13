import cv2
import numpy as np
import os

# Scale: 10 cm total canvas, 5.5 cm inner ArUco
canvas_size = 1000
aruco_size = 550  

canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

# Generate DICT_5X5_1000, ID: 0
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, aruco_size)

# Center it to create the margins
offset = (canvas_size - aruco_size) // 2
canvas[offset:offset+aruco_size, offset:offset+aruco_size] = marker_image

# Draw the 1.1 cm radius circles in the corners
circle_radius = 110  
margin_center = offset // 2  

centers = [
    (margin_center, margin_center),
    (canvas_size - margin_center, margin_center),
    (margin_center, canvas_size - margin_center),
    (canvas_size - margin_center, canvas_size - margin_center)
]

for center in centers:
    cv2.circle(canvas, center, circle_radius, 0, -1)

# Save directly to the local Desktop
save_path = os.path.join(os.path.expanduser("~"), "Desktop", "custom_10cm_id0.png")
success = cv2.imwrite(save_path, canvas)

if success:
    print(f"Target locked and generated! Saved to: {save_path}")
else:
    print("Failed to save. Check folder permissions.")
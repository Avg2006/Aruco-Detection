import cv2
import numpy as np


cam_data = []
dist_data =[]
rect_data =[]
proj_data =[]

#---------------------------------------CUSTOMISATION--------------------------------------------------------------------
CHECKERBOARD = (6, 8) # Define the dimensions of the checkerboard (inner corners)
side_length = 27.3 #side length of each square in mm

# Setup termination criteria for the subpixel algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

threedpoints = []
twodpoints = []

objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*side_length

# Change number to change camera
cap = cv2.VideoCapture(1)

print("Press 's' to capture a frame for calibration.")
print("Press 'c' to finish capturing and run the calibration.")
print("Press 'q' to quit without calibrating.")

images_captured = 0
grayColor = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Is your camera being used by another app?")
        break

    cv2.imshow('Live Camera Feed', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        found, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            threedpoints.append(objectp3d)

            # THIS IS THE SUB-PIXEL REFINEMENT 
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)

            images_captured += 1
            print(f"Success! Captured frame {images_captured}. Keep moving the board and press 's' again.")
            
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, found)
            cv2.imshow('Live Camera Feed', frame)
            cv2.waitKey(500) 
        else:
            print("Couldn't find the complete chessboard in that frame. Try adjusting lighting or the angle.")

    elif key == ord('c'):
        if images_captured > 10: 
            print(f"\nStarting calibration with {images_captured} images... Please wait.")
            break
        else:
            print(f"You only have {images_captured} images. It's recommended to have at least 10. Press 's' to take more, or 'c' again to force it.")
            # We don't break here so you can keep capturing if you want!

    elif key == ord('q'):
        print("Quitting without calibrating.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if len(twodpoints) > 0:
    # 'ret' here is the crucial RMS Re-projection Error!
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None
    )

    print(f"\n--- CALIBRATION RESULTS ---")
    print(f"RMS Re-projection Error: {ret:.4f}")
    
    # Grade the sub-pixel accuracy
    if ret < 0.25:
        print("STATUS: EXCELLENT. Sub-pixel accuracy achieved. You are ready for tracking!")
    elif ret < 0.5:
        print("STATUS: GOOD. Standard accuracy achieved, but tracking might drift slightly.")
    else:
        print("STATUS: POOR. RMS error is too high. You need to recapture slower and steadier.")

    # Get the image dimensions
    h, w = grayColor.shape

    # --- 1. Rectification Matrix ---
    rectification_matrix = np.eye(3)

    # --- 2. Projection Matrix ---
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 0, (w,h))
    projection_matrix = np.zeros((3, 4))
    projection_matrix[:, :3] = new_camera_matrix

    # Convert arrays to flat, comma-separated strings
    cam_data = ", ".join(map(str, matrix.flatten().tolist()))
    dist_data = ", ".join(map(str, distortion.flatten().tolist()))
    rect_data = ", ".join(map(str, rectification_matrix.flatten().tolist()))
    proj_data = ", ".join(map(str, projection_matrix.flatten().tolist()))

    print("\nCopy the block below directly into your default_cam.yaml file:\n")
    print("------------------------------")
    
    yaml_output = f"""image_width: {w}
image_height: {h}
camera_name: default_cam
camera_matrix:
  rows: 3
  cols: 3
  data: [{cam_data}]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [{dist_data}]
rectification_matrix:
  rows: 3
  cols: 3
  data: [{rect_data}]
projection_matrix:
  rows: 3
  cols: 4
  data: [{proj_data}]"""

    print(yaml_output)
    print("------------------------------")
    
else:
    print("\nNo valid images were captured. Calibration aborted.")
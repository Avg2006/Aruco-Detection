import cv2
import numpy as np
import yaml

#---------------------------------------CUSTOMISATION--------------------------------------------------------------------
CHECKERBOARD = (6, 8)          # Inner corners (same as your intrinsic script)
side_length_mm = 27.3          # Side length of each square in mm
CAMERA_INFO_YAML = "/home/zak/.ros/camera_info/default_cam.yaml"   # Path to the yaml your intrinsic script produced
CAMERA_INDEX = 1                # Change number to change camera

# ROS/TF wants meters, so convert here -- everything downstream (world_xyz,
# the launch file, TF) stays in meters.
side_length = side_length_mm / 1000.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * side_length

#---------------------------------------LOAD INTRINSICS--------------------------------------------------------------------

with open(CAMERA_INFO_YAML, "r") as f:
    cam_yaml = yaml.safe_load(f)

camera_matrix = np.array(cam_yaml["camera_matrix"]["data"]).reshape(3, 3)
distortion = np.array(cam_yaml["distortion_coefficients"]["data"])

print(f"Loaded intrinsics from {CAMERA_INFO_YAML}")

#---------------------------------------HELPERS--------------------------------------------------------------------

def quaternion_from_matrix(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return qx, qy, qz, qw


def solve_and_report(gray, corners):
    """Runs solvePnP, returns (rvec, tvec, mean_reproj_error_px)."""
    success, rvec, tvec = cv2.solvePnP(objectp3d, corners, camera_matrix, distortion)
    if not success:
        return None, None, None

    reproj, _ = cv2.projectPoints(objectp3d, rvec, tvec, camera_matrix, distortion)
    err = np.linalg.norm(reproj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    return rvec, tvec, err.mean()


#---------------------------------------LIVE CAPTURE LOOP--------------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

print("\nTape/place the checkerboard wherever you want world (0,0,0) to be.")
print("Press 's' to try a pose solve on the current frame.")
print("Press 'c' to CONFIRM the last successful solve and print the result.")
print("Press 'q' to quit without saving.\n")

last_rvec, last_tvec, last_err = None, None, None

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

        if not found:
            print("Couldn't find the complete chessboard in that frame. Try adjusting lighting or the angle.")
            continue

        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)

        rvec, tvec, mean_err = solve_and_report(grayColor, corners2)
        if rvec is None:
            print("solvePnP failed on that frame, try again.")
            continue

        last_rvec, last_tvec, last_err = rvec, tvec, mean_err

        debug_frame = frame.copy()
        cv2.drawChessboardCorners(debug_frame, CHECKERBOARD, corners2, found)
        cv2.drawFrameAxes(debug_frame, camera_matrix, distortion, rvec, tvec, side_length * 3)
        cv2.imshow('Live Camera Feed', debug_frame)
        cv2.waitKey(500)

        print(f"Pose solved. Mean reprojection error: {mean_err:.4f} px", end="  ")
        if mean_err < 0.5:
            print("(EXCELLENT)")
        elif mean_err < 1.0:
            print("(GOOD)")
        else:
            print("(POOR -- recommend recapturing: steadier angle, better lighting, less blur)")
        print("Press 'c' to confirm this result, or 's' again to retry.\n")

    elif key == ord('c'):
        if last_rvec is None:
            print("No successful capture yet -- press 's' first.")
            continue

        # rvec/tvec = T_cam_board (world/board -> camera). Invert for world -> cam.
        R_cam_board, _ = cv2.Rodrigues(last_rvec)
        T_cam_board = np.eye(4)
        T_cam_board[:3, :3] = R_cam_board
        T_cam_board[:3, 3] = last_tvec.flatten()

        T_world_cam = np.linalg.inv(T_cam_board)
        R_world_cam = T_world_cam[:3, :3]
        t_world_cam = T_world_cam[:3, 3]
        qx, qy, qz, qw = quaternion_from_matrix(R_world_cam)

        print("\n--- CONFIRMED EXTRINSIC RESULT ---")
        print(f"Mean reprojection error: {last_err:.4f} px")
        print("\nPaste into multi_cam_launch.py CAMERAS[...] entry:\n")
        print("------------------------------")
        print(f'"world_xyz": [{t_world_cam[0]:.5f}, {t_world_cam[1]:.5f}, {t_world_cam[2]:.5f}],')
        print(f'"world_quat": [{qx:.5f}, {qy:.5f}, {qz:.5f}, {qw:.5f}],')
        print("------------------------------")
        break

    elif key == ord('q'):
        print("Quitting without saving.")
        break

cap.release()
cv2.destroyAllWindows()
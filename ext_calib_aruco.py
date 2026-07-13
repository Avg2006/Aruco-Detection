import cv2
import numpy as np
import yaml

#---------------------------------------CUSTOMISATION--------------------------------------------------------------------
CAMERA_INFO_YAML = "/home/zak/.ros/camera_info/default_cam2.yaml"   # yaml produced by your intrinsic calibration script
CAMERA_INDEX = 3                        # /dev/videoN index for this camera

ARUCO_DICT = "DICT_5X5_250"             # must match what aruco_node.py uses
marker_size = 0.08664                   # meters -- side length of the marker's black square
# No fixed marker ID -- whichever marker you show first becomes the world
# origin. See TARGET_MARKER_ID logic in the capture loop below.

#---------------------------------------LOAD INTRINSICS--------------------------------------------------------------------

with open(CAMERA_INFO_YAML, "r") as f:
    cam_yaml = yaml.safe_load(f)

camera_matrix = np.array(cam_yaml["camera_matrix"]["data"]).reshape(3, 3)
distortion = np.array(cam_yaml["distortion_coefficients"]["data"])
print(f"Loaded intrinsics from {CAMERA_INFO_YAML}")

# Same corner ordering as your aruco_node.py object_points
half_size = marker_size / 2.0
object_points = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
], dtype=np.float32)

dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

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
    return np.array([qx, qy, qz, qw])


def average_quaternions(quats):
    """Simple averaging via eigen-decomposition of the outer-product sum --
    robust to sign ambiguity (q and -q represent the same rotation)."""
    quats = np.array(quats)
    # flip sign so all quats are in the same hemisphere as the first one
    for i in range(1, len(quats)):
        if np.dot(quats[0], quats[i]) < 0:
            quats[i] = -quats[i]
    M = quats.T @ quats
    eigvals, eigvecs = np.linalg.eigh(M)
    avg = eigvecs[:, np.argmax(eigvals)]
    if avg[3] < 0:  # keep qw positive by convention
        avg = -avg
    return avg


#---------------------------------------LIVE CAPTURE LOOP--------------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)

print(f"\nShow any ArUco marker to use as the world origin.")
print("Press 's' to capture a pose sample from the current frame (repeat several times).")
print("  - First successful capture picks the marker ID (or asks you if several are visible).")
print("  - Later captures only count if they see that SAME marker ID.")
print("Press 'c' to CONFIRM and average all captured samples into a final result.")
print("Press 'r' to reset/discard all captured samples AND unlock the marker ID choice.")
print("Press 'q' to quit without saving.\n")

captured_t = []   # list of (x, y, z) world_cam translations
captured_q = []   # list of quaternions world_cam
TARGET_MARKER_ID = None   # locked in on first successful capture

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Is your camera being used by another app?")
        break

    cv2.imshow('Live Camera Feed', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            print("No markers visible in this frame. Try again.")
            continue

        visible_ids = ids.flatten().tolist()

        if TARGET_MARKER_ID is None:
            # First successful capture -- decide which marker becomes the origin.
            if len(visible_ids) == 1:
                TARGET_MARKER_ID = visible_ids[0]
                print(f"Using marker ID {TARGET_MARKER_ID} as world origin (auto-selected).")
            else:
                print(f"Multiple markers visible: {sorted(set(visible_ids))}")
                choice = input("Type the marker ID to use as world origin: ").strip()
                try:
                    TARGET_MARKER_ID = int(choice)
                except ValueError:
                    print("Not a valid integer, try 's' again.")
                    continue
                if TARGET_MARKER_ID not in visible_ids:
                    print(f"Marker {TARGET_MARKER_ID} isn't one of the visible markers, try 's' again.")
                    TARGET_MARKER_ID = None
                    continue

        if TARGET_MARKER_ID not in visible_ids:
            print(f"Marker {TARGET_MARKER_ID} not visible in this frame (locked target). Try again.")
            continue

        idx = visible_ids.index(TARGET_MARKER_ID)
        corner = corners[idx]

        success, rvec, tvec = cv2.solvePnP(object_points, corner, camera_matrix, distortion)
        if not success:
            print("solvePnP failed on that frame, try again.")
            continue

        reproj, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        err = np.linalg.norm(reproj.reshape(-1, 2) - corner.reshape(-1, 2), axis=1).mean()

        # rvec/tvec = T_cam_marker (world/marker -> camera). Invert for world -> cam.
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        T_cam_marker = np.eye(4)
        T_cam_marker[:3, :3] = R_cam_marker
        T_cam_marker[:3, 3] = tvec.flatten()
        T_world_cam = np.linalg.inv(T_cam_marker)

        t_world_cam = T_world_cam[:3, 3]
        q_world_cam = quaternion_from_matrix(T_world_cam[:3, :3])

        captured_t.append(t_world_cam)
        captured_q.append(q_world_cam)

        debug_frame = frame.copy()
        cv2.aruco.drawDetectedMarkers(debug_frame, [corner], np.array([[TARGET_MARKER_ID]]))
        cv2.drawFrameAxes(debug_frame, camera_matrix, distortion, rvec, tvec, marker_size * 1.5)
        cv2.imshow('Live Camera Feed', debug_frame)
        cv2.waitKey(300)

        print(f"Sample {len(captured_t)} captured. Reprojection error: {err:.4f} px", end="  ")
        if err < 0.5:
            print("(EXCELLENT)")
        elif err < 1.5:
            print("(OK)")
        else:
            print("(POOR -- move closer / reduce angle / improve lighting, this sample will still be averaged in)")

    elif key == ord('r'):
        captured_t = []
        captured_q = []
        TARGET_MARKER_ID = None
        print("Discarded all samples and unlocked marker choice. Starting fresh.\n")

    elif key == ord('c'):
        if len(captured_t) == 0:
            print("No samples captured yet -- press 's' first.")
            continue
        if len(captured_t) < 5:
            print(f"Only {len(captured_t)} sample(s) -- recommend at least 5-10 for a stable average. "
                  f"Press 'c' again to force it anyway, or 's' to capture more.")

        t_arr = np.array(captured_t)
        t_final = np.median(t_arr, axis=0)   # median = robust to occasional bad frame
        q_final = average_quaternions(captured_q)

        print(f"\n--- CONFIRMED EXTRINSIC RESULT ({len(captured_t)} samples averaged) ---")
        print("\nPaste into multi_cam_launch.py CAMERAS[...] entry:\n")
        print("------------------------------")
        print(f'"world_xyz": [{t_final[0]:.5f}, {t_final[1]:.5f}, {t_final[2]:.5f}],')
        print(f'"world_quat": [{q_final[0]:.5f}, {q_final[1]:.5f}, {q_final[2]:.5f}, {q_final[3]:.5f}],')
        print("------------------------------")
        break

    elif key == ord('q'):
        print("Quitting without saving.")
        break

cap.release()
cv2.destroyAllWindows()
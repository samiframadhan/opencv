import cv2 as cv
from cv2 import aruco
import numpy as np

# dictionary to specify type of the marker
marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# detect the marker
param_markers = aruco.DetectorParameters()

# utilizes default camera/webcam driver
cap = cv.VideoCapture(0)

aruco_detector = aruco.ArucoDetector(marker_dict, param_markers)

calibration_file = 'camera_parameters2.yaml'
# loading camera parameters
cv_file = cv.FileStorage(calibration_file, cv.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_coeff = cv_file.getNode("dist_coeff").mat()
cv_file.release()

def estimate_pose_single_marker(marker_corners, marker_length, camera_matrix, dist_coeff):
    marker_points = np.array([
        [-marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, marker_length / 2, 0],
        [marker_length / 2, -marker_length / 2, 0],
        [-marker_length / 2, -marker_length / 2, 0],
    ], dtype=np.float32)
    
    rvecs, tvecs, rejected = [], [], []
    for corners in marker_corners:
        reject, R, t = cv.solvePnP(
            marker_points,
            corners,
            camera_matrix,
            distCoeffs=dist_coeff, useExtrinsicGuess=False, flags=cv.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        rejected.append(reject)
    return rvecs, tvecs, rejected

# iterate through multiple frames, in a live video feed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # turning the frame to grayscale-only (for efficiency)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco_detector.detectMarkers(
        gray_frame
    )
    # Estimate distance to each marker
    if marker_corners:
        # rvecs, tvecs, _ = cv.estimatePoseSingleMarkers(
        #     marker_corners,
        #     0.05,
        #     camera_matrix,
        #     dist_coeff,
        # )
        rvecs, tvecs, _ = estimate_pose_single_marker(
            marker_corners,
            0.025,
            camera_matrix,
            dist_coeff,
        )
        
        for i in range(len(marker_IDs)):
            cv.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeff,
                rvecs[i],
                tvecs[i],
                0.03,
            )
            
        cv.putText(
            frame,
            f"rvecs: {np.array2string(rvecs[0].flatten(), precision=2)}",
            (10, 30),
            cv.FONT_HERSHEY_PLAIN,
            1.2,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
        # distance display
        cv.putText(
            frame,
            f"tvecs: {np.array2string(tvecs[0].flatten(), precision=2)}",
            (10, 60),
            cv.FONT_HERSHEY_PLAIN,
            1.2,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )
    # getting conrners of markers
    # if marker_corners:
    #     for ids, corners in zip(marker_IDs, marker_corners):
    #         cv.polylines(
    #             frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
    #         )
    #         corners = corners.reshape(4, 2)
    #         corners = corners.astype(int)
    #         top_right = corners[0].ravel()
    #         top_left = corners[1].ravel()
    #         bottom_right = corners[2].ravel()
    #         bottom_left = corners[3].ravel()
    #         cv.putText(
    #             frame,
    #             f"id: {ids[0]}",
    #             top_right,
    #             cv.FONT_HERSHEY_PLAIN,
    #             1.3,
    #             (200, 100, 0),
    #             2,
    #             cv.LINE_AA,
    #         )
            # print(ids, "  ", corners)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
import numpy as np
import cv2






#1000, 0, 320], [0, 1000, 240], [0, 0, 1]
def arUco_camera_calib_distance(real_marker_size_mm,image):
    # Estimate pose of each detected marker
    markerLength = real_marker_size_mm  # mm
    camera_matrix = np.array([
    [3024,    0, 2016],
    [   0, 3024, 1512],
    [   0,    0,    1]
], dtype=np.float32)
    
    dist_coeffs = np.array([[ 0.0, -0.0, 0.0, 0.0, 0]], dtype=np.float32)
    

    
    # Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)



    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerLength, camera_matrix, dist_coeffs
    )

    z_values = [tvec[0][2] for tvec in tvecs]
    mean_z = np.mean(z_values)
    
    return mean_z


image_path = "New Tests/test.jpg"  

image = cv2.imread(image_path)

print ( arUco_camera_calib_distance(30,image))
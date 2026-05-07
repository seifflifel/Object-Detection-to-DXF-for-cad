
import time
import cv2
import numpy as np
import os
import torch
from mobile_sam import sam_model_registry, SamPredictor
from PIL import Image
import subprocess
from grab_phone_screen import ScreenGrabber
import tkinter as tk
from tkinter import ttk
import win32gui

os.chdir("C:/CP files")  # change to your working directory

# Function to get list of open windows
def get_open_windows():
    windows = []
    def enum_windows(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).strip()
            if title and title not in ["Program Manager", ""]:  # Filter out system/empty titles
                windows.append(title)
    win32gui.EnumWindows(enum_windows, None)
    return sorted(windows)  # Sort for consistency

def select_window_title():
    root = tk.Tk()
    root.title("Select Window and Camera Distance")
    root.geometry("400x220")

    selected_title = tk.StringVar()
    camera_distance = tk.StringVar(value="130")  # Default value

    label = tk.Label(root, text="Select phone screen app to capture:")
    label.pack(pady=5)

    dropdown = ttk.Combobox(root, textvariable=selected_title, state="readonly", width=50)
    dropdown.pack(pady=5)

    def refresh_dropdown():
        windows = get_open_windows()
        if windows:
            dropdown["values"] = windows
            if not selected_title.get() or selected_title.get() not in windows:
                dropdown.set(windows[0])  # Default to first window
        else:
            dropdown["values"] = ["No windows found"]
            dropdown.set("No windows found")
        root.after(15000, refresh_dropdown)  # Refresh every 15 sec

    refresh_dropdown()

    distance_label = tk.Label(root, text="Enter camera distance (mm):")
    distance_label.pack(pady=5)
    distance_entry = tk.Entry(root, textvariable=camera_distance)
    distance_entry.pack(pady=5)

    def confirm_selection():
        title = selected_title.get()
        if title and title != "No windows found":
            windows = get_open_windows()
            if title in windows:
                try:
                    distance = float(camera_distance.get())
                    root.selected_title = title
                    root.camera_distance = distance
                    root.destroy()
                except ValueError:
                    tk.messagebox.showerror("Invalid Input", "Camera distance must be a number.")
            else:
                tk.messagebox.showerror("Error", f"Window '{title}' not found. Please try again.")
        else:
            tk.messagebox.showwarning("Warning", "No valid window selected. Please try again.")

    confirm_button = tk.Button(root, text="Confirm", command=confirm_selection)
    confirm_button.pack(pady=10)

    root.mainloop()
    return getattr(root, "selected_title", None), getattr(root, "camera_distance", None)

def capture_app_window(grabber):  # Add grabber parameter
    return grabber.capture_frame()

def visualize_aruco_live(grabber,camera_distance):  # Add grabber parameter
    aruco_dict_type = cv2.aruco.DICT_4X4_50

    # Load the dictionary and detector parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    while True:
        frame = capture_app_window(grabber)  # Pass grabber to capture_app_window
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("ArUco on Mirrored Screen click 'c' and enter", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Main Pipeline
            cv2.imwrite("camera.png", frame)  # save the current frame
            dim = detect_and_segment("camera.png", camera_distance, aruco_dict, parameters, detector, corners, ids)  # camera distance 130 in mm
            invert_image()
            convert_to_bmp()
            vectorize_with_potrace(dim, "DXF/Dxf real measure.dxf")
            print("✅ Full processing done.")
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def chessboard_calibration():
    return 0

def ArUco_calib(path, camera_distance, aruco_dict, parameters, detector, corners, ids):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError("Image not found.")

    # ---- Define YOUR specific IDs in YOUR order ----
    desired_ids = [0, 2, 3, 1]  # TL, TR, BR, BL

    # Build mapping from ID -> corners
    id_to_corners = {id_: corners[i][0] for i, id_ in enumerate(ids.flatten())}

    # Marker corners for alignment
    src_pts = np.array([
        id_to_corners[0][3],  # top-left marker, its corner 
        id_to_corners[2][0],  # top-right marker, 
        id_to_corners[3][1],  # bottom-right marker, 
        id_to_corners[1][2],  # bottom-left marker, 
    ], dtype=np.float32)

    # Here we'll make the output a cropped fronto-parallel rect
    width = int(max(
        np.linalg.norm(src_pts[1] - src_pts[0]),
        np.linalg.norm(src_pts[2] - src_pts[3])
    ))
    height = int(max(
        np.linalg.norm(src_pts[3] - src_pts[0]),
        np.linalg.norm(src_pts[2] - src_pts[1])
    ))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # ---- Compute homography and warp ----
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, H, (width, height))
    
    cv2.imwrite("images/warped.png", warped)
    # cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # marker_corners: list of 4x2 arrays for each marker (after warping)
    real_marker_size_mm = 30.0
    x_dists = []
    y_dists = []

    # marker IDs are 0, 2, 1, 3
    marker_ids = [0, 2, 1, 3]

    # id_to_corners: {id: 4x2 array}
    marker_corners = [id_to_corners[mid] for mid in marker_ids]

    for corners in marker_corners:
        # X distances (horizontal)
        x_dists.append(np.linalg.norm(corners[0] - corners[3]))
        x_dists.append(np.linalg.norm(corners[1] - corners[2]))
        # Y distances (vertical)
        y_dists.append(np.linalg.norm(corners[0] - corners[1]))
        y_dists.append(np.linalg.norm(corners[3] - corners[2]))
    # Calculate average distances
    avg_x = np.mean(x_dists)
    avg_y = np.mean(y_dists)

    # 130 mm
    avg_z = camera_distance
    x_scale = real_marker_size_mm / avg_x
    y_scale = real_marker_size_mm / avg_y
    T = [x_scale, y_scale, warped, avg_z]
    print(f"X scale absolute : {x_scale} mm/pixel, Y scale absolute: {y_scale} mm/pixel, Avg Z: {avg_z} mm")
    return T

def detect_and_segment(path, camera_distance, aruco_dict, parameters, detector, corners, ids):
    T = ArUco_calib(path, camera_distance, aruco_dict, parameters, detector, corners, ids)

    image = T[2]
    if image is None:
        raise FileNotFoundError("Image not found.")
    clone = image.copy()

    height, width, _ = image.shape
    dim = [width * T[0], height * T[1]]

    print("original width :", width)
    print("original height :", height)
    print("width scaled :", dim[0])
    print("height scaled :", dim[1])
    width_scale = dim[0] / width
    height_scale = dim[1] / height
    print("width scale : ", width_scale, " height scale : ", height_scale)  # use for solid
    output_file = "width_scale.txt"
    with open(output_file, 'w') as f:
        f.write(str(width_scale))

    sam_checkpoint = "mobile_sam.pt"
    model_type = "vit_t"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    predictor = SamPredictor(sam)
    image_bgr = T[2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    clicked = []
    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 1:
            clicked.append((x, y))
            cv2.circle(image_bgr, (x, y), 5, (0, 0, 255), -1)
            cv2.namedWindow("Click on the Object and click 'enter'", cv2.WINDOW_NORMAL)
            cv2.imshow("Click on the Object and click 'enter' ", image_bgr)
            #cv2.resizeWindow("Click on the Object and click 'enter'", 1080 * (int)(width / height), 1080)

    cv2.namedWindow("Click on the Object and click 'enter'", cv2.WINDOW_NORMAL)
    cv2.imshow("Click on the Object and click 'enter'", image_bgr)
    #cv2.resizeWindow("Click on the Object and click 'enter'", 1080 * (int)(width / height), 1080)
    cv2.setMouseCallback("Click on the Object and click 'enter'", click_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not clicked:
        raise Exception("No point clicked")

    input_point = np.array([clicked[0]])
    input_label = np.array([1])
    masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
    best_mask = masks[np.argmax(scores)]
    mask_display = image_rgb.copy()
    mask_display[best_mask] = (255, 0, 0)
    cv2.namedWindow("Segmented Object", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmented Object", cv2.cvtColor(mask_display, cv2.COLOR_RGB2BGR))
    #cv2.resizeWindow("Segmented Object", 1080 * (int)(width / height), 1080)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("images/mask.png", best_mask.astype(np.uint8) * 255)

    return dim

def invert_image():
    image = Image.open("images/mask.png")
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    data = np.array(image)
    rgb = data[:, :, :3]
    alpha = data[:, :, 3]
    inverted_rgb = np.where(alpha[:, :, None] > 0, 255 - rgb, rgb)
    inverted_rgb[alpha == 0] = [255, 255, 255]
    new_data = np.dstack((inverted_rgb, alpha))
    inverted_image = Image.fromarray(new_data, mode="RGBA")
    background = Image.new("RGBA", inverted_image.size, (255, 255, 255, 255))
    result = Image.alpha_composite(background, inverted_image)
    result.save("images/black_contour_white_bg.png")

def convert_to_bmp():
    input_path = "images/black_contour_white_bg.png"
    output_path = "images/bitmap.bmp"
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {input_path}. Check the file path.")
    img_rgb = img.convert("RGB")
    img_rgb.save(output_path, format="BMP")

def vectorize_with_potrace(dim, output_dxf):
    input_bmp = "images/bitmap.bmp"
    
    potrace_path = os.path.join(os.getcwd(), "potrace-1.16.win64", "potrace.exe")

    potrace_cmd = [
        potrace_path,
        input_bmp,
        "-b", "dxf",
        "-o", output_dxf,
        "-a", "1",
        "-t", "1",
        "-O", "1",
        "--width", str(dim[0]),
        "--height", str(dim[1]),
    ]
    try:
        subprocess.run(potrace_cmd, check=True)
    except subprocess.CalledProcessError as e:
        exit(1)
    print("DONE!")

def main():
    #run scrcpy
    subprocess.Popen(["C:\\CP files\\scrcpy-win64-v3.3.1\\scrcpy.exe"])
    time.sleep(5)

    # Get window title from user
    window_title, camera_distance = select_window_title()
    if not window_title or camera_distance is None:
        print("❌ No valid window or camera distance selected. Exiting.")
        exit(1)

    # Initialize ScreenGrabber with selected title
    grabber = ScreenGrabber(window_title)
    visualize_aruco_live(grabber,camera_distance)  # Pass grabber to visualize_aruco_live

if __name__ == "__main__":
    main()
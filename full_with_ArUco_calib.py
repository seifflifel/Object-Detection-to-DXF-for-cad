import cv2
import numpy as np
import os
import torch
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from datetime import datetime
import subprocess

os.chdir("C:/Users/Seifo/Documents/Stage ete 2025")
#x 36.5 42.3 y 41.3 
input_path =""
output_path =""
blur_amount = 0
real_distance = 10.0


def select_ruler_points_with_matplotlib(image_path):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title('Zoom/pan, then click two points (close window when done)')
    clicked_points = []
    def onclick(event):
        if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar is not None:
            if fig.canvas.toolbar.mode != '':
                return
        if event.inaxes != ax:
            return
        if len(clicked_points) < 2:
            x, y = int(event.xdata), int(event.ydata)
            clicked_points.append((x, y))
            ax.plot(x, y, 'go')
            fig.canvas.draw()
        if len(clicked_points) == 2:
            plt.close(fig)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return clicked_points if len(clicked_points) == 2 else None

def ArUco_calib():
    image_path = "arucoTests/test5.jpg"  # Make sure this file exists!
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

# Detect ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image)

# Draw detected markers for visual check
    debug_img = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
    cv2.imshow('Detected Markers', debug_img)
    cv2.waitKey(0)

# ---- Define YOUR specific IDs in YOUR order ----
    desired_ids = [0, 2, 3, 1]  # TL, TR, BR, BL

# Build mapping from ID -> corners
    id_to_corners = {id_: corners[i][0] for i, id_ in enumerate(ids.flatten())}


# Take CORNER 0 (top-left corner of each marker) for alignment
    src_pts = np.array([
    id_to_corners[0][3],  # top-left marker, its corner 
    id_to_corners[2][0],  # top-right marker, 
    id_to_corners[3][1],  # bottom-right marker, 
    id_to_corners[1][2],  # bottom-left marker, 
], dtype=np.float32)

# ---- Define the destination rectangle ----
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
    
    cv2.imwrite("warped.png",warped)
    cv2.imshow("Warped", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# marker_corners: list of 4x2 arrays for each marker (after warping)
    real_marker_size_mm = 30.0
    x_dists = []
    y_dists = []

# Example: your marker IDs are 0, 2, 1, 3 (in any order you want)
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


    avg_x = np.mean(x_dists)
    avg_y = np.mean(y_dists)

    x_scale = real_marker_size_mm / avg_x
    y_scale = real_marker_size_mm / avg_y
    T =[x_scale,y_scale,warped]
    print(f"X scale: {x_scale} mm/pixel, Y scale: {y_scale} mm/pixel")
    return T

def detect_and_segment():
    real_distance_mm = 10.0
    T  = ArUco_calib()
    image = T[2]
    cv2.imshow("Warped", T[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if image is None:
        raise FileNotFoundError("Image not found.")
    clone = image.copy()
    
    
    height,width,_ = image.shape
    dim = [ width * T[0] , height*T[1]]
    print("height :",height)
    print("width :",width)
    print("width scaled",dim[0])
    print("height scaled",dim[1])


    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
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
            cv2.imshow("Click on the background", image_bgr)

    cv2.imshow("Click on the background", image_bgr)
    cv2.setMouseCallback("Click on the background", click_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not clicked:
        raise Exception("No point clicked")

    input_point = np.array([clicked[0]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    mask_display = image_rgb.copy()
    mask_display[best_mask] = (0, 255, 0)
    cv2.imshow("Segmented Object", cv2.cvtColor(mask_display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("mask.png", best_mask.astype(np.uint8) * 255)

    return dim

def extract_contour():
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    inverted_mask = mask
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(transparent_image, [contour], -1, color=(0, 0, 0, 255), thickness=2)
    cv2.imwrite('contour_transparent_black.png', transparent_image)


def smooth_contour():
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    inverted_mask = mask
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    alpha_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(alpha_mask, contours, -1, color=255, thickness=2)

    def make_rgba_from_alpha(alpha):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0:3] = 255
        rgba[:, :, 3] = alpha
        return rgba

    blur_kernel = 1
    iteration_counter = 1
    preview_window = 'Feathered Contour'

    def apply_blur(k):
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        blurred_alpha = cv2.GaussianBlur(alpha_mask, (k, k), 0)
        return blurred_alpha

    def update(val):
        nonlocal blur_kernel
        blur_kernel = cv2.getTrackbarPos('Blur Size', preview_window)
        blurred_alpha = apply_blur(blur_kernel)
        preview = cv2.merge([blurred_alpha]*3)
        cv2.imshow(preview_window, preview)

    def save_image():
        nonlocal iteration_counter
        blurred_alpha = apply_blur(blur_kernel)
        rgba_image = make_rgba_from_alpha(blurred_alpha)
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"contour_{date_str}_{iteration_counter:02d}.png"
        cv2.imwrite(filename, rgba_image)
        iteration_counter += 1

    cv2.namedWindow(preview_window)
    cv2.createTrackbar('Blur Size', preview_window, 1, 50, update)
    update(1)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_image()
        elif key == ord('q') or key == 27:
            break
    cv2.destroyAllWindows()

def stack_image():
    img = cv2.imread('contour_transparent_black.png', cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('Could not load image! Check the path.')
    b, g, r, a = cv2.split(img)
    alpha_float = a.astype(np.float32) / 255.0
    stack_count = 20
    accum_alpha = alpha_float * stack_count
    accum_alpha = np.clip(accum_alpha, 0, 1.0)
    new_alpha = (accum_alpha * 255).astype(np.uint8)
    result = cv2.merge([255*np.ones_like(new_alpha),
                        255*np.ones_like(new_alpha),
                        255*np.ones_like(new_alpha),
                        new_alpha])
    cv2.imwrite('contour_stacked_white.png', result)

def invert_image():
    image = Image.open("contour_stacked_white.png")
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
    result.save("black_contour_whitebg.png")

def convert_to_bmp():
    input_path = "black_contour_whitebg.png"
    output_path = "black_contour_whitebg_pillow_conversion.bmp"
    try:
        img = Image.open(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {input_path}. Check the file path.")
    img_rgb = img.convert("RGB")
    img_rgb.save(output_path, format="BMP")

def vectorize_with_potrace(dim):
    input_bmp = "black_contour_whitebg_pillow_conversion.bmp"
    output_dxf = "test5.dxf"
    
    potrace_cmd = [
        r"C:\\Users\\Seifo\\Documents\\Stage ete 2025\\potrace-1.16.win64\\potrace.exe",
        input_bmp,
        "-b", "dxf",
        "-o", output_dxf,
        "-a","1",
        "-t","1",
        "-O","1",
        "--width", str(dim[0]),
        "--height", str(dim[1]),
        
    ]
    try:
        subprocess.run(potrace_cmd, check=True)
    except subprocess.CalledProcessError as e:
        exit(1)
    print("DONE!")




def main():
    dim = detect_and_segment()
    extract_contour()
    smooth_contour()
    stack_image()
    invert_image()
    convert_to_bmp()
    vectorize_with_potrace(dim)

if __name__ == "__main__":
    main()
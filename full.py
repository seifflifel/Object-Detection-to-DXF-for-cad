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


def detect_and_segment():
    real_distance_mm = 10.0
    image_path = "Newer batch crop.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found.")
    clone = image.copy()

    # Try matplotlib-based point selection first
    clicked_points = select_ruler_points_with_matplotlib(image_path)
    if not clicked_points:
        print("Matplotlib selection failed or cancelled. Reverting to OpenCV selection.")
        clicked_points = []
        def click_ruler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
                clicked_points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Click two ruler points", image)
        cv2.imshow("Click two ruler points", image)
        cv2.setMouseCallback("Click two ruler points", click_ruler)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if len(clicked_points) != 2:
        raise ValueError("You must select exactly 2 ruler points.")

    pt1, pt2 = clicked_points
    pixel_distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
    scale = real_distance_mm / pixel_distance 
    height,width,_ = image.shape
    dim = [ height*scale , width*scale ]
    print("height :",height)
    print("width :",width)
    print("height scaled",dim[0])
    print("width scaled",dim[1])


    sam_checkpoint = "sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    predictor = SamPredictor(sam)
    image_bgr = cv2.imread("Newer batch crop.jpg")
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
    output_dxf = "black_contour_whitebg.dxf"
    x= dim[0]
    potrace_cmd = [
        r"C:\\Users\\Seifo\\Documents\\Stage ete 2025\\potrace-1.16.win64\\potrace.exe",
        input_bmp,
        "-b", "dxf",
        "-o", output_dxf,
        "-a","10000",
        "-t","10000",
        "-O","10000",
        "--width", str(dim[0]),
        "--height", str(dim[1]),
        
    ]
    try:
        subprocess.run(potrace_cmd, check=True)
    except subprocess.CalledProcessError as e:
        exit(1)




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
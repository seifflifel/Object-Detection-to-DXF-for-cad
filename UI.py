import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from grabscrcpycph2349screenclass import ScreenGrabber

#

# App state
click_coords = []
image_width, image_height = 720, 1000
frame_count = 0

class MainApp:
    def __init__(self, root):
        
        self.grabber = ScreenGrabber("camera.png")  # Adjust window title as needed

        self.root = root
        self.root.title("Image Processing GUI")

        # Main frame to hold left (canvas) and right (controls) sections
        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # === Left: Live Image Preview ===
        self.canvas = tk.Canvas(self.main_frame, width=image_width, height=image_height, bg="black")
        self.canvas.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nw")
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Placeholder image
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.update_image()

        # === Right: Controls Frame ===
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nw")

        # Snapshot & Confirm Buttons
        self.snapshot_button = ttk.Button(self.controls_frame, text="📸 Snapshot", command=self.on_snapshot)
        self.snapshot_button.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        self.confirm_button = ttk.Button(self.controls_frame, text="✅ Confirm Selection", command=self.on_confirm)
        self.confirm_button.grid(row=1, column=0, sticky="ew", padx=10, pady=5)

        # Status Text
        self.status_text = tk.StringVar()
        self.status_text.set("Click on the object to segment it...")
        self.status_label = ttk.Label(self.controls_frame, textvariable=self.status_text)
        self.status_label.grid(row=2, column=0, padx=10, pady=5)

        # Collapsible Calibration Panel
        self.calib_frame = ttk.LabelFrame(self.controls_frame, text="🔧 Calibration Panel", relief=tk.RIDGE)
        self.calib_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(self.calib_frame, text="Rows:").grid(row=0, column=0, sticky="e")
        self.rows_entry = ttk.Entry(self.calib_frame)
        self.rows_entry.insert(0, "6")
        self.rows_entry.grid(row=0, column=1)

        ttk.Label(self.calib_frame, text="Columns:").grid(row=1, column=0, sticky="e")
        self.cols_entry = ttk.Entry(self.calib_frame)
        self.cols_entry.insert(0, "9")
        self.cols_entry.grid(row=1, column=1)

        ttk.Label(self.calib_frame, text="Square Size (mm):").grid(row=2, column=0, sticky="e")
        self.size_entry = ttk.Entry(self.calib_frame)
        self.size_entry.insert(0, "25.0")
        self.size_entry.grid(row=2, column=1)

        ttk.Button(self.calib_frame, text="📸 Capture Calibration Image",
                   command=self.on_calib_capture).grid(row=3, column=0, columnspan=2, pady=5)
        ttk.Button(self.calib_frame, text="🎯 Run Calibration",
                   command=self.on_run_calibration).grid(row=4, column=0, columnspan=2, pady=5)

    def update_image(self):
        try:
            frame = self.grabber.capture_frame()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            self.tk_image = ImageTk.PhotoImage(img_pil)
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)
        except Exception as e:
            print("Error grabbing frame:", e)

        self.root.after(50, self.update_image)


    def on_canvas_click(self, event):
        global click_coords
        click_coords = (event.x, event.y)
        self.status_text.set(f"Clicked at: {click_coords}")

    def on_snapshot(self):
        self.status_text.set("Snapshot captured!")

    def on_confirm(self):
        self.status_text.set(f"Confirmed click at: {click_coords}")

    def on_calib_capture(self):
        self.status_text.set("Calibration image captured.")

    def on_run_calibration(self):
        rows = self.rows_entry.get()
        cols = self.cols_entry.get()
        size = self.size_entry.get()
        self.status_text.set(f"Running calibration with {rows}x{cols}, {size}mm squares.")

# Run the GUI
if __name__ == "__main__":
    #launch()  # Start scrcpy or any other required process
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
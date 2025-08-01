import cv2
from PIL import ImageGrab
import win32gui
import numpy as np
import ctypes
from time import sleep

class ScreenGrabber:
    def __init__(self, window_title):
        self.window_title = window_title.lower()
        self.hwnd = None
        self.position = None
        self._find_window()

    def _find_window(self):
        sleep(1)  # Optional: Give time for window to appear
        ctypes.windll.user32.SetProcessDPIAware()

        windowslist = []

        def enum_windows(hwnd, _):
            wintext = win32gui.GetWindowText(hwnd)
            if wintext.strip():
                windowslist.append((hwnd, wintext))

        win32gui.EnumWindows(enum_windows, None)

        print(f"Searching for window: '{self.window_title}'")
        for hwnd, title in windowslist:
            print(f"Checking: '{title}'")
            if self.window_title in title.lower():
                print(f"✅ Found window: '{title}' (HWND: {hwnd})")
                self.hwnd = hwnd
                self.position = win32gui.GetWindowRect(hwnd)
                return

        raise Exception(f"❌ No window with title containing '{self.window_title}' was found.")

    def capture_frame(self):
        if not self.position:
            raise Exception("Window position not available.")

        screenshot = ImageGrab.grab(bbox=self.position)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def start_preview(self):
        if not self.position:
            raise Exception("Window position not available.")

        print(f"📷 Previewing window at position: {self.position}")
        while True:
            frame = self.capture_frame()
            cv2.imshow("Screen Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

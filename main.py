import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from utilities import get_parking_spots, checkBlockStatus


class ParkingSpotGUI:
    def __init__(self, mask_path, video_path):
        self.mask = cv2.imread(mask_path, 0)
        self.cap = cv2.VideoCapture(video_path)
        self.connected_components = cv2.connectedComponentsWithStats(self.mask, 4, cv2.CV_32S)
        self.spots = get_parking_spots(self.connected_components)
        self.spots_status = [None for _ in self.spots]
        self.diffs = [None for _ in self.spots]
        self.previous_frame = None
        self.frame_number = 0
        self.step = 20

        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.aspect_ratio = self.video_width / self.video_height

        # Set canvas width based on aspect ratio
        self.canvas_width = 800
        self.canvas_height = int(self.canvas_width / self.aspect_ratio)

        self.root = tk.Tk()
        self.root.title("Parking Spot Video Display")

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.update()

    def calc_diff(self, im1, im2):
        return np.abs(np.mean(im1) - np.mean(im2))

    def update(self):
        ret, frame = self.cap.read()

        if not ret:
            self.root.quit()
            return

        if self.frame_number % self.step == 0 and self.previous_frame is not None:
            for spot_indx, spot in enumerate(self.spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                self.diffs[spot_indx] = self.calc_diff(spot_crop, self.previous_frame[y1:y1 + h, x1:x1 + w, :])

        if self.frame_number % self.step == 0:
            if self.previous_frame is None:
                arr_ = range(len(self.spots))
            else:
                arr_ = [j for j in np.argsort(self.diffs) if self.diffs[j] / np.amax(self.diffs) > 0.4]
            for spot_indx in arr_:
                spot = self.spots[spot_indx]
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status = checkBlockStatus(spot_crop)
                self.spots_status[spot_indx] = spot_status

        if self.frame_number % self.step == 0:
            self.previous_frame = frame.copy()

        for spot_indx, spot in enumerate(self.spots):
            spot_status = self.spots_status[spot_indx]
            x1, y1, w, h = spot

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 255, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(self.spots_status)), str(len(self.spots_status))),
                    (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Resize frame to fit the canvas
        frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if cv2.waitKey(25) & 0xFF == ord('q'):
            self.root.quit()

        self.frame_number += 1
        self.root.after(25, self.update)



if __name__ == "__main__":
    app = ParkingSpotGUI('ParkingLotVideo/mask_1920_1080.png', 'ParkingLotVideo/parking_1920_1080_loop.mp4')
    status_window = tk.Tk()
    status_window.title("Parking Spot Status") # Update status window periodically
    status_window.mainloop()


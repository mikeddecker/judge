import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class VideoLabeler:
    def __init__(self, root, video_path, display_width=800, display_height=600):
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.display_width = display_width
        self.display_height = display_height
        
        self.frame_label = tk.Label(root)
        self.frame_label.pack()

        self.slider = ttk.Scale(root, from_=0, to=self.cap.get(cv2.CAP_PROP_FRAME_COUNT), orient=tk.HORIZONTAL, command=self.slider_changed)
        self.slider.pack(fill=tk.X, expand=1)
        
        self.play_button = tk.Button(root, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=10)
        
        self.pause_button = tk.Button(root, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.select_start_button = tk.Button(root, text="Border Start", command=self.select_start)
        self.select_start_button.pack(side=tk.LEFT, padx=10)
        
        self.select_end_button = tk.Button(root, text="Border End", command=self.select_end)
        self.select_end_button.pack(side=tk.LEFT, padx=10)
        
        self.selected_start_frame = None
        self.selected_end_frame = None
        
        self.selected_ranges = []  # List to store selected ranges
        
        self.playing = True
        self.show_frame()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def show_frame(self):
        if self.playing:
            ret, frame = self.cap.read()
            if ret:
                frame = self.resize_frame(frame)
                self.display_frame(frame)
                current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.slider.set(current_frame)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # (loop video)
        
        self.root.after(100, self.show_frame)
        
    def resize_frame(self, frame):
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        if width > self.display_width or height > self.display_height:
            if width > height:
                new_width = self.display_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = self.display_height
                new_width = int(new_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        return frame
        
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame_label.imgtk = imgtk
        self.frame_label.configure(image=imgtk)
        
    def slider_changed(self, value):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(value)))
        ret, frame = self.cap.read()
        if ret:
            frame = self.resize_frame(frame)
            self.display_frame(frame)

    def play_video(self):
        self.playing = True

    def pause_video(self):
        self.playing = False
    
    def select_start(self):
        self.selected_start_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    def select_end(self):
        self.selected_end_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.selected_start_frame is not None and self.selected_end_frame is not None:
            self.selected_ranges.append((self.selected_start_frame, self.selected_end_frame))
            self.selected_start_frame = None
            self.selected_end_frame = None
            print(f"Selected range: {self.selected_ranges[-1]}")

    def on_closing(self):
        self.playing = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    video_path = '/home/miked/code/judge/videos/20240201_atelier_005.mp4'  # Replace this with your video file path
    root = tk.Tk()
    app = VideoLabeler(root, video_path, display_width=1200, display_height=900)  # Adjust display width and height as needed
    root.mainloop()

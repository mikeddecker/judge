import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class VideoLabeler:
    def __init__(self, root, video_path, display_width=800, display_height=600):
        self.root = root
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.display_width = display_width
        self.display_height = display_height
        self.current_pos = 0
        
        self.frame_label = tk.Label(root)
        self.frame_label.pack()

        # Custom slider using Canvas
        self.slider_canvas = tk.Canvas(root, height=30, bg='white')
        self.slider_canvas.pack(fill=tk.X, expand=1)
        self.slider_canvas.bind('<Button-1>', self.slider_clicked)
        self.slider_canvas.bind('<B1-Motion>', self.slider_dragged)
        self.slider_canvas.bind('<MouseWheel>', self.slider_zoomed)
        
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
        self.zoom_factor = 1.0  # Initial zoom factor
        self.visible_range_start = 0  # Start of the visible range
        self.visible_range_end = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # End of the visible range

        # Key bindings
        self.root.bind('<s>', self.select_start_key)
        self.root.bind('<n>', self.next_frame)
        self.root.bind('<e>', self.select_end_key)
        self.root.bind('<z>', self.zoom_in_key)
        self.root.bind('<o>', self.zoom_out_key)
        self.root.bind('<q>', self.on_closing)
        
        self.show_frame()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def show_frame(self):
        ret, frame = self.cap.read()
        self.current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)

        if ret:
            frame = self.resize_frame(frame)
            self.display_frame(frame)
            self.update_slider()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # (loop video)
        
        if self.playing:
            self.root.after(30, self.show_frame)
        
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

    def slider_clicked(self, event):
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        visible_frames = self.visible_range_end - self.visible_range_start
        self.current_pos = int((event.x / self.slider_canvas.winfo_width()) * visible_frames + self.visible_range_start)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)
        ret, frame = self.cap.read()
        if ret:
            frame = self.resize_frame(frame)
            self.display_frame(frame)
        self.update_slider()

    def slider_dragged(self, event):
        self.slider_clicked(event)
        
    def slider_zoomed(self, event):
        if event.delta > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, 10.0)
        elif event.delta < 0:
            self.zoom_factor = max(self.zoom_factor / 1.1, 1.0)
        self.adjust_visible_range()
        self.update_slider()

    def zoom_in_key(self, event):
        self.zoom_factor = min(self.zoom_factor * 1.1, 10.0)
        self.adjust_visible_range()
        self.update_slider()
    
    def zoom_out_key(self, event):
        self.zoom_factor = max(self.zoom_factor / 1.1, 1.0)
        self.adjust_visible_range()
        self.update_slider()
        
    def adjust_visible_range(self):
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        visible_frames = total_frames / self.zoom_factor
        center_frame = self.current_pos
        
        # Calculate the start and end of the visible range to center the current frame
        self.visible_range_start = max(center_frame - visible_frames // 2, 0)
        self.visible_range_end = min(center_frame + visible_frames // 2, total_frames)
        
        # Adjust current position if out of new visible range
        if self.current_pos < self.visible_range_start:
            self.current_pos = self.visible_range_start
        if self.current_pos > self.visible_range_end:
            self.current_pos = self.visible_range_end
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_pos)

    def update_slider(self):
        self.slider_canvas.delete("all")
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        visible_frames = self.visible_range_end - self.visible_range_start
        slider_pos = ((self.current_pos - self.visible_range_start) / visible_frames) * self.slider_canvas.winfo_width()
        
        # Draw slider background
        self.slider_canvas.create_rectangle(0, 0, self.slider_canvas.winfo_width(), 30, fill='white')
        
        # Draw selected ranges
        for start, end in self.selected_ranges:
            if end > self.visible_range_start and start < self.visible_range_end:
                start_pos = max((start - self.visible_range_start) / visible_frames, 0) * self.slider_canvas.winfo_width()
                end_pos = min((end - self.visible_range_start) / visible_frames, 1) * self.slider_canvas.winfo_width()
                self.slider_canvas.create_rectangle(start_pos, 0, end_pos, 30, fill='lightblue')

        # Draw current range
        if self.selected_start_frame is not None and self.selected_end_frame is None:
            start_pos = max((self.selected_start_frame - self.visible_range_start) / visible_frames, 0) * self.slider_canvas.winfo_width()
            end_pos = min((self.cap.get(cv2.CAP_PROP_POS_FRAMES) - self.visible_range_start) / visible_frames, 1) * self.slider_canvas.winfo_width()
            self.slider_canvas.create_rectangle(start_pos, 0, end_pos, 30, fill='lightgreen')
        
        # Draw current position marker
        self.slider_canvas.create_line(slider_pos, 0, slider_pos, 30, fill='red')

    def play_video(self):
        self.playing = True
        self.show_frame()

    def pause_video(self):
        self.playing = False

    def next_frame(self, event):
        if not self.playing:
            self.show_frame()

    def select_start_key(self, event):
        self.select_start()

    def select_end_key(self, event):
        self.select_end()

    def select_start(self):
        self.selected_start_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    def select_end(self):
        self.selected_end_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.selected_start_frame is not None and self.selected_end_frame is not None:
            self.selected_ranges.append((self.selected_start_frame, self.selected_end_frame))
            self.selected_start_frame = None
            self.selected_end_frame = None
            print(f"Selected range: {self.selected_ranges[-1]}")

    def on_closing(self, event=None):
        self.playing = False
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    video_path = '/home/miked/code/judge/videos/20240201_atelier_005.mp4'  # Replace this with your video file path
    root = tk.Tk()
    app = VideoLabeler(root, video_path, display_width=1200, display_height=900)  # Adjust display width and height as needed
    root.mainloop()

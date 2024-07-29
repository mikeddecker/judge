import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from DataRepository import DataRepository
from DataGeneratorFrames import DataGeneratorRectangles

# -------------------------- DEFINING GLOBAL VARIABLES -------------------------
# these are just the colors we'll be using to paint our rooms (frames)
selectionbar_color = '#eff5f6'
sidebar_color = '#F5E1FD'
header_color = '#53366b'
visualisation_frame_color = "#ffffff"


class VideoLabeler:
    def __init__(self, root, video_id, batch_size, predict=False):
        # --- Video config & Y values ---
        video_root_folder = '/media/miked/Elements/Judge/FINISHED-DB-READY/'
        self.repo = DataRepository()
        self.video_id = video_id
        self.video_path = video_root_folder + self.repo.get_path(video_id)
        self.y_frames = self.repo.query_framelabels(video_id) # frame list: rectangle, skillstatus
        self.y_skills = self.repo.get_borders(video_id)  # Skill-list: toad, start = 100, end = 120

        if predict:
            # TODO : load y predicted values from models.
            pass

        # --- Prepare screen ---

        # --- BASIC APP LAYOUT ---
        self.root = root
        self.root.title(self.video_path)
        self.root.geometry("1100x700")
        self.root.config(background=selectionbar_color)
        # icon = tk.PhotoImage(file='images\\WU_logo.png') # Icon for header on top
        # self.iconphoto(True, icon)

        # --- HEADER ---
        self.header = tk.Frame(root, bg=header_color)
        self.header.place(relx=0.2, rely=0, relwidth=0.8, relheight=0.04)

        # --- SIDEBAR ---
        # SIDEBAR FRAME
        self.sidebar = tk.Frame(root, bg=sidebar_color)
        self.sidebar.place(relx=0, rely=0, relwidth=0.2, relheight=1)

        # -- Mainframe ---
        self.main_frame = tk.Frame(root)
        self.main_frame.place(relx=0.2, rely=0.04, relwidth=0.8, relheight=0.84)

        # --- Footer ---
        # Slider & buttons
        self.footer = tk.Frame(root)
        self.footer.place(relx=0.2, rely=0.88, relwidth=0.8, relheight=0.12)


        # --- Loading video & stats---
        self.cap = cv2.VideoCapture(self.video_path)
        self.original_width = self.cap.get(3)  # float `width`
        self.original_height = self.cap.get(4)  # float `height`
        self.update_framesize()
        

        self.current_frameNr = 0
        
        self.rectangles = True

        if self.rectangles:
            self.current_modus = 'watch'
            self.create_readonly_textbox(self.current_modus)
            self.text_box_functions = {
                'center' : self.move_rectangle,
                'resize' : self.resize_rectangle,
                'watch' : lambda x, y: None,
            }
            self.frame_label = tk.Label(self.main_frame)
            self.frame_label.pack()
            self.frame_label.bind("<Button-1>", self.on_click)

        first_frame = self.y_frames.loc[0]
        print(first_frame)
        print((0.5 if first_frame['rect_center_x'] is None else first_frame['rect_center_x']))
        self.rect_center_x = int((0.5 if first_frame['rect_center_x'] is None else first_frame['rect_center_x']) * self.original_width)
        self.rect_center_y = int((0.5 if first_frame['rect_center_y'] is None else first_frame['rect_center_y']) * self.original_height)
        self.rect_size = 1 if first_frame['rect_size'] is None else first_frame['rect_size']

        # Custom slider using Canvas
        self.slider_canvas = tk.Canvas(self.footer, height=30, bg='white')
        self.slider_canvas.pack(fill=tk.X, expand=1)
        self.slider_canvas.bind('<Button-1>', self.slider_clicked)
        self.slider_canvas.bind('<B1-Motion>', self.slider_dragged)
        self.slider_canvas.bind('<MouseWheel>', self.slider_zoomed)
        
        self.play_button = tk.Button(self.footer, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=10)
        
        self.pause_button = tk.Button(self.footer, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.select_start_button = tk.Button(self.footer, text="Border Start", command=self.select_start)
        self.select_start_button.pack(side=tk.LEFT, padx=10)
        
        self.select_end_button = tk.Button(self.footer, text="Border End", command=self.select_end)
        self.select_end_button.pack(side=tk.LEFT, padx=10)
        
        self.selected_start_frame = None
        self.selected_end_frame = None
        

        self.playing = True
        self.zoom_factor = 1.0  # Initial zoom factor
        self.visible_range_start = 0  # Start of the visible range
        self.visible_range_end = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)  # End of the visible range

        # Key bindings
        self.root.bind('<s>', self.select_start_key)
        self.root.bind('<n>', self.next_frame)
        self.root.bind('<p>', self.previous_frame)
        self.root.bind('<e>', self.select_end_key)
        self.root.bind('<z>', self.zoom_in_key)
        self.root.bind('<o>', self.zoom_out_key)
        self.root.bind('<q>', self.on_closing)
        self.root.bind('<r>', self.remove_border)
        self.root.bind('<t>', self.toggle_textbox)
        self.root.bind('<y>', self.reset_rectangle)
        self.root.bind('<space>', self.toggle_play_pause)
        root.bind("<Configure>", lambda event: self.update_framesize())

        
        if self.rectangles:
            self.playing = False
        
        self.show_frame()


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_framesize(self):
        self.root.update()
        prop_relx = self.main_frame.winfo_width() / self.original_width
        prop_rely = self.main_frame.winfo_height() / self.original_height
        smallest_rel_size = min(prop_relx, prop_rely)
        self.display_height = int(smallest_rel_size * self.original_height)
        self.display_width = int(smallest_rel_size * self.original_width)
    
    def show_frame(self):
        ret, frame = self.cap.read()
        self.current_frameNr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            if self.current_modus == 'watch':
                self.follow_rectangle()
            else:
                self.update_framelabels_rectangles()
            self.add_rectangle_to_frame(frame)
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
        
    def add_rectangle_to_frame(self, frame):
        width = min(self.original_height, self.original_width)
        xmin = int(self.rect_center_x - (self.rect_size * width / 2))
        ymin = int(self.rect_center_y - (self.rect_size * width / 2))
        xmax = int(xmin + self.rect_size * width)
        ymax = int(ymin + self.rect_size * width)

        xmin = max(5, xmin)
        ymin = max(5, ymin)

        wanted_frame = np.array(frame[ymin:ymax, xmin:xmax])    
        frame[ymin-4:ymax+5, xmin-5:xmax+5] = [250,0,0] # y, x
        frame[ymin:ymax, xmin:xmax] = wanted_frame

        # Dot
        frame[self.rect_center_y-10:self.rect_center_y+10, self.rect_center_x-10:self.rect_center_x+10] = [0,0,255]
        return frame

    def on_click(self, event):
        # Get the size of the displayed image, it is slightly off with the given size
        disp_width, disp_height = self.frame_label.winfo_width(), self.frame_label.winfo_height()
        scale_x = self.original_width / disp_width
        scale_y = self.original_height / disp_height
        original_x = int(event.x * scale_x)
        original_y = int(event.y * scale_y)

        print(f"Clicked at ({event.x}, {event.y}), corresponding to ({original_x}, {original_y}) in original frame")

        self.get_textbox_function()(original_x, original_y)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frameNr - 1)

        if not self.playing:
            self.show_frame()

        
    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.frame_label.imgtk = imgtk
        self.frame_label.configure(image=imgtk)

    def slider_clicked(self, event):
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        visible_frames = self.visible_range_end - self.visible_range_start
        self.current_frameNr = int((event.x / self.slider_canvas.winfo_width()) * visible_frames + self.visible_range_start)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frameNr)
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
        center_frame = self.current_frameNr
        
        # Calculate the start and end of the visible range to center the current frame
        self.visible_range_start = max(center_frame - visible_frames // 2, 0)
        self.visible_range_end = min(center_frame + visible_frames // 2, total_frames)
        
        # Adjust current position if out of new visible range
        if self.current_frameNr < self.visible_range_start:
            self.current_frameNr = self.visible_range_start
        if self.current_frameNr > self.visible_range_end:
            self.current_frameNr = self.visible_range_end
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frameNr)

    def update_slider(self):
        """Updates the coloring of the slider"""
        self.slider_canvas.delete("all")
        total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        visible_frames = self.visible_range_end - self.visible_range_start
        slider_pos = ((self.current_frameNr - self.visible_range_start) / visible_frames) * self.slider_canvas.winfo_width()
        
        # Draw slider background
        self.slider_canvas.create_rectangle(0, 0, self.slider_canvas.winfo_width(), 30, fill='white')
        
        # Draw selected ranges
        for idx, row in self.y_skills[['frame_start', 'frame_end']].iterrows():
            start = row['frame_start']
            end = row['frame_end']
            
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

    def toggle_play_pause(self, event=None):
        self.playing = not self.playing
        if self.playing:
            self.show_frame()

    def next_frame(self, event):
        if not self.playing:
            self.show_frame()

    def previous_frame(self, event):
        if not self.playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frameNr - 2)
            self.show_frame()

    def select_start_key(self, event):
        self.select_start()

    def select_end_key(self, event):
        self.select_end()

    def select_start(self):
        self.selected_start_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.selected_end_frame = None
    
    def select_end(self):
        self.selected_end_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if self.selected_start_frame is not None and self.selected_end_frame is not None:
            if self.repo.is_valid_border(self.video_id, self.selected_start_frame, self.selected_end_frame):
                print('valid')
                idx = len(self.y_skills)
                self.repo.add_border(self.video_id, self.selected_start_frame, self.selected_end_frame, 1)
                self.y_skills.loc[idx] = [self.video_id, self.selected_start_frame, self.selected_end_frame, 1, 0]
                self.selected_start_frame = None
                self.selected_end_frame = None
                print(f"Selected range: {2}")
            else:
                print('invalid')
                self.selected_start_frame = None
                self.selected_end_frame = None

    def remove_border(self, event):
        df_skill = self.y_skills[(self.y_skills['frame_start'] <= self.current_frameNr) & (self.current_frameNr <= self.y_skills['frame_end'])]
        if len(df_skill) > 0:
            self.repo.remove_border(self.video_id, df_skill.iloc[0]['frame_start'], df_skill.iloc[0]['frame_end'])
            self.y_skills = self.repo.get_borders(self.video_id)        

    def on_closing(self, event=None):
        self.playing = False

        self.repo.uninserted_borders_to_framelabels(self.video_id)
        
        print('done')
        
        self.cap.release()
        self.root.destroy()

    def create_readonly_textbox(self, initial_text):
        text_box = tk.Text(self.header, wrap='word', height=1)
        text_box.insert('1.0', initial_text)  # Insert the initial text at the beginning
        text_box.config(state='disabled')    # Make the text box read-only
        text_box.pack()
        self.text_box = text_box

    def update_textbox(self, new_text):
        self.text_box.config(state='normal')   # Temporarily make the text box editable
        self.text_box.delete('1.0', tk.END)    # Delete the current text
        self.text_box.insert('1.0', new_text)  
        self.text_box.config(state='disabled')
    
    def toggle_textbox(self, event):
        match self.current_modus:
            case 'center':
                self.current_modus = 'resize'
            case 'resize':
                self.current_modus = 'watch'
            case 'watch':
                self.current_modus = 'center'
            case _:
                self.current_modus = 'watch'
        self.update_textbox(self.current_modus)

    def get_textbox_function(self):
        return self.text_box_functions[self.current_modus]
    
    def is_non_or_nan(self, x):
        return x is None or np.isnan(x)
    
    def reset_rectangle(self, event):
        print('reset borders from', self.rect_center_x, self.rect_center_y, self.rect_size)
        self.rect_center_x = int(0.5 * self.original_width)
        self.rect_center_y = int(0.5 * self.original_height)
        self.rect_size = 1
        print('borders resetted', self.rect_center_x, self.rect_center_y, self.rect_size)

    def follow_rectangle(self):
        # just needs two params
        print('current pos: ', self.current_frameNr)
        curr_frame = self.y_frames.loc[self.current_frameNr-1]
        if not self.is_non_or_nan(curr_frame['rect_size']):
            self.rect_center_x = int(curr_frame['rect_center_x'] * self.original_width)
            self.rect_center_y = int(curr_frame['rect_center_y'] * self.original_height)
            self.rect_size = curr_frame['rect_size']


    def move_rectangle(self, x, y):
        print('rectangle moved', x, y)
        self.rect_center_x = int(x)
        self.rect_center_y = int(y)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frameNr - 1)
    
    def resize_rectangle(self, x, y):
        print('rectangle resized')
        rel_dist_x = abs(x - self.rect_center_x) / self.original_width
        rel_dist_y = abs(y - self.rect_center_y) / self.original_height
        grens_x = abs(self.rect_size * min(self.original_width, self.original_height) / 2) / self.original_width
        grens_y = abs(self.rect_size * min(self.original_width, self.original_height) / 2) / self.original_height
        if rel_dist_x < grens_x and rel_dist_y < grens_y:
            self.rect_size *= 0.97
        else:
            self.rect_size *= 1.03
    
    def update_framelabels_rectangles(self):
        print('update label ', self.current_frameNr)
        rel_x = self.rect_center_x / self.original_width
        rel_y = self.rect_center_y / self.original_height
        self.y_frames.loc[self.current_frameNr-1, 'rect_center_x'] = rel_x
        self.y_frames.loc[self.current_frameNr-1, 'rect_center_y'] = rel_y
        self.y_frames.loc[self.current_frameNr-1, 'rect_size'] = self.rect_size
        self.repo.update_rectangle(self.video_id, self.current_frameNr, rel_x, rel_y, self.rect_size)

if __name__ == "__main__":
    watch_predictions = False
    prediction_model = '../models/frames_skillborder_CNN_model_96pixels_history.pkl'
    video_id = 16
    batch_size = 9999
    root = tk.Tk()
    app = VideoLabeler(root, video_id, batch_size)  # Adjust display width and height as needed
    root.mainloop()

    import tkinter as tk
    from tkinter import filedialog, messagebox
    from tkinter import ttk
    import threading
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
    import imutils
    
    # Constants
    INPAINT_RADIUS = 20
    TEXT_DETECTION_INTERVAL = 10  # Text detection ko kam frequently karein
    
    class VideoEditorApp(tk.Tk):
        def check_cuda_support(self):
            try:
                test_gpu_mat = cv2.cuda_GpuMat()
                return True
            except cv2.error:
                return False
    
        def __init__(self):
            super().__init__()
            self.title("Video Editor – Preview & Remove Logo/Text")
            self.geometry("800x650")
            self.input_video_path = ""
            self.output_video_path = ""
            self.preview_image = None
            self.manual_logo_boxes = []
            self.is_selecting_logo = False
            self.start_x, self.start_y = None, None
            self.rect_ids = []
            self.preview_canvas = None
            self.current_frame_for_preview = None
            self.logo_trackers = []
            self.initial_text_boxes = []  # Initial text boxes ko store karein
    
            self.use_gpu = self.check_cuda_support()
            print("GPU support:", "Enabled" if self.use_gpu else "Disabled")
    
            self.create_widgets()
    
        def create_widgets(self):
            btn_frame = tk.Frame(self)
            btn_frame.pack(pady=10)
    
            self.upload_button = tk.Button(btn_frame, text="Upload Video", command=self.upload_video)
            self.upload_button.grid(row=0, column=0, padx=5)
    
            self.select_logo_button = tk.Button(btn_frame, text="Select Logo Area(s)", command=self.start_manual_logo_select, state=tk.DISABLED)
            self.select_logo_button.grid(row=0, column=1, padx=5)
    
            self.clear_logos_button = tk.Button(btn_frame, text="Clear Logo Selections", command=self.clear_manual_logos, state=tk.DISABLED)
            self.clear_logos_button.grid(row=0, column=2, padx=5)
    
            self.save_button = tk.Button(btn_frame, text="Select Save Location", command=self.select_save_location, state=tk.DISABLED)
            self.save_button.grid(row=0, column=3, padx=5)
    
            self.process_button = tk.Button(btn_frame, text="Process Video", command=self.start_processing, state=tk.DISABLED)
            self.process_button.grid(row=0, column=4, padx=5)
    
            self.preview_canvas = tk.Canvas(self, width=640, height=360, bg="gray")
            self.preview_canvas.pack(pady=10)
            self.status_label = tk.Label(self, text="Waiting", font=("Arial", 14))
            self.status_label.pack(pady=10)
            self.progress = ttk.Progressbar(self, orient='horizontal', length=600, mode='determinate')
            self.progress.pack(pady=10)
    
        def upload_video(self):
            path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
            if path:
                self.input_video_path = path
                self.status_label.config(text="Video Uploaded")
                self.manual_logo_boxes = []
                self.initial_text_boxes = []
                self.rect_ids = []
                self.select_logo_button.config(state=tk.NORMAL)
                self.clear_logos_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.NORMAL)
                self.show_preview()
    
        def select_save_location(self):
            path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if path:
                self.output_video_path = path
                self.status_label.config(text="Save Location Selected")
    
        def clear_manual_logos(self):
            self.manual_logo_boxes = []
            for rect_id in self.rect_ids:
                self.preview_canvas.delete(rect_id)
            self.rect_ids = []
            self.status_label.config(text="Manual Logo Selections Cleared.")
            self.clear_logos_button.config(state=tk.DISABLED)
            self.show_preview()
    
        def show_preview(self):
            cap = cv2.VideoCapture(self.input_video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                messagebox.showerror("Error", "Could not read video frame.")
                return
    
            self.current_frame_for_preview = frame.copy()
            self.initial_text_boxes = self.detect_text_elements(frame)  # Initial text detection
            logo_boxes = self.manual_logo_boxes
    
            preview_frame = self.preview_frame_with_boxes(frame, self.initial_text_boxes, logo_boxes)
    
            frame_height, frame_width = frame.shape[:2]
            canvas_width, canvas_height = 640, 360
            aspect_ratio = frame_width / float(frame_height)
            if aspect_ratio > canvas_width / canvas_height:
                preview_width = canvas_width
                preview_height = int(canvas_width / aspect_ratio)
            else:
                preview_height = canvas_height
                preview_width = int(canvas_height * aspect_ratio)
    
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(preview_rgb).resize((preview_width, preview_height), Image.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(pil_image)
    
            self.preview_canvas.config(width=preview_width, height=preview_height)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_image)
    
            self.redraw_manual_logo_rectangles(preview_width, preview_height, frame_width, frame_height)
    
            if self.manual_logo_boxes:
                self.status_label.config(text=f"Logos Selected: {len(self.manual_logo_boxes)}. Click 'Process Video'.")
            else:
                self.status_label.config(text="Preview Ready. Select logos or process for text removal.")
            self.process_button.config(state=tk.NORMAL)
    
        def redraw_manual_logo_rectangles(self, preview_width, preview_height, original_width, original_height):
            for rect_id in self.rect_ids:
                self.preview_canvas.delete(rect_id)
            self.rect_ids = []
            for logo_box in self.manual_logo_boxes:
                x_orig, y_orig, w_orig, h_orig = logo_box
                x_preview = int(x_orig * (preview_width / original_width))
                y_preview = int(y_orig * (preview_height / original_height))
                w_preview = int(w_orig * (preview_width / original_width))
                h_preview = int(h_orig * (preview_height / original_height))
                rect_id = self.preview_canvas.create_rectangle(x_preview, y_preview, x_preview + w_preview, y_preview + h_preview,
                                                               outline='red', width=2)
                self.rect_ids.append(rect_id)
    
        def start_manual_logo_select(self):
            self.status_label.config(text="Click and drag to select logo area(s).")
            self.is_selecting_logo = True
            self.select_logo_button.config(state=tk.DISABLED)
            self.clear_logos_button.config(state=tk.DISABLED)
            self.preview_canvas.bind("<ButtonPress-1>", self.on_mouse_press)
            self.preview_canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.preview_canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
    
        def on_mouse_press(self, event):
            if not self.is_selecting_logo: return
            self.start_x, self.start_y = event.x, event.y
    
        def on_mouse_drag(self, event):
            if not self.is_selecting_logo or self.start_x is None: return
            cur_x, cur_y = event.x, event.y
            self.preview_canvas.delete("temp_rect")
            self.preview_canvas.create_rectangle(self.start_x, self.start_y, cur_x, cur_y, outline='red', width=2, tags="temp_rect")
    
        def on_mouse_release(self, event):
            if not self.is_selecting_logo or self.start_x is None: return
            self.is_selecting_logo = False
            self.select_logo_button.config(state=tk.NORMAL)
            self.clear_logos_button.config(state=tk.NORMAL)
    
            x1, y1 = self.start_x, self.start_y
            x2, y2 = event.x, event.y
            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
    
            self.preview_canvas.delete("temp_rect")
            if (x_max - x_min) > 5 and (y_max - y_min) > 5:
                original_width, original_height = self.current_frame_for_preview.shape[1], self.current_frame_for_preview.shape[0]
                preview_width, preview_height = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
    
                x_orig = int(x_min * (original_width / preview_width))
                y_orig = int(y_min * (original_height / preview_height))
                w_orig = int((x_max - x_min) * (original_width / preview_width))
                h_orig = int((y_max - y_min) * (original_height / preview_height))
    
                self.manual_logo_boxes.append((x_orig, y_orig, w_orig, h_orig))
                self.show_preview()
            else:
                self.status_label.config(text="Selection too small. Try again.")
    
            self.preview_canvas.unbind("<ButtonPress-1>")
            self.preview_canvas.unbind("<B1-Motion>")
            self.preview_canvas.unbind("<ButtonRelease-1>")
    
        def start_processing(self):
            if not self.input_video_path or not self.output_video_path:
                messagebox.showerror("Error", "Upload video and select save location.")
                return
    
            self.upload_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.process_button.config(state=tk.DISABLED)
            self.select_logo_button.config(state=tk.DISABLED)
            self.clear_logos_button.config(state=tk.DISABLED)
            self.status_label.config(text="Processing: 0%")
            self.progress["value"] = 0
    
            thread = threading.Thread(target=self.process_video)
            thread.start()
    
        def process_video(self):
            cap = cv2.VideoCapture(self.input_video_path)
            if not cap.isOpened():
                self.update_status("Error opening video file")
                return
    
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
    
            frame_number = 0
            self.logo_trackers = []
    
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
    
                # Text detection sirf pehle frame ya interval par
                text_boxes = self.initial_text_boxes if frame_number % TEXT_DETECTION_INTERVAL != 0 else self.detect_text_elements(frame)
    
                # Logo tracking with MOSSE
                if frame_number == 0:
                    for box in self.manual_logo_boxes:
                        tracker = cv2.legacy.TrackerMOSSE_create()
                        tracker.init(frame, box)
                        self.logo_trackers.append((tracker, box))
                else:
                    logo_boxes_frame = []
                    for i, (tracker, last_box) in enumerate(self.logo_trackers):
                        success, box = tracker.update(frame)
                        logo_boxes_frame.append(box if success else last_box)
                        if success:
                            self.logo_trackers[i] = (tracker, box)
                edited_frame = self.remove_elements_from_frame(frame, text_boxes, logo_boxes_frame)
                out.write(edited_frame)
    
                frame_number += 1
                progress_percent = int((frame_number / total_frames) * 100)
                self.update_progress(progress_percent)
    
            cap.release()
            out.release()
            self.update_status("Processing Complete!")
            self.upload_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.process_button.config(state=tk.NORMAL)
            self.select_logo_button.config(state=tk.NORMAL)
            self.clear_logos_button.config(state=tk.NORMAL)
    
        def detect_text_elements(self, frame):
            net = cv2.dnn.readNet("frozen_east_text_detection.pb")
            resized = imutils.resize(frame, width=320)
            ratio_h, ratio_w = frame.shape[0] / float(resized.shape[0]), frame.shape[1] / float(resized.shape[1])
    
            blob = cv2.dnn.blobFromImage(resized, 1.0, (resized.shape[1], resized.shape[0]), (123.68, 116.78, 103.94), swapRB=True)
            net.setInput(blob)
            (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
    
            rects, confidences = [], []
            num_rows, num_cols = scores.shape[2:4]
            for y in range(num_rows):
                scores_data = scores[0, 0, y]
                x_data0, x_data1, x_data2, x_data3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
                for x in range(num_cols):
                    if scores_data[x] < 0.5:
                        continue
                    offset_x, offset_y = x * 4.0, y * 4.0
                    h, w = x_data0[x] + x_data2[x], x_data1[x] + x_data3[x]
                    rects.append((int(offset_x), int(offset_y), int(w), int(h)))
                    confidences.append(scores_data[x])
    
            boxes = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)
            text_boxes = [(int(r[0] * ratio_w), int(r[1] * ratio_h), int(r[2] * ratio_w), int(r[3] * ratio_h)) for r in [rects[i] for i in boxes]]
            return text_boxes
    
        def preview_frame_with_boxes(self, frame, text_boxes, logo_boxes):
            preview = frame.copy()
            for (x, y, w, h) in text_boxes:
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y, w, h) in logo_boxes:
                cv2.rectangle(preview, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            return preview
    
        def remove_elements_from_frame(self, frame, text_boxes, logo_boxes):
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for (x, y, w, h) in text_boxes + logo_boxes:
                x, y, w, h = int(x), int(y), int(w), int(h)
                mask[y:y+h, x:x+w] = 255
    
            if self.use_gpu:
                try:
                    gpu_frame = cv2.cuda_GpuMat(); gpu_frame.upload(frame)
                    gpu_mask = cv2.cuda_GpuMat(); gpu_mask.upload(mask)
                    inpainted_gpu = cv2.cuda.inpaint(gpu_frame, gpu_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
                    return inpainted_gpu.download()
                except:
                    return cv2.inpaint(frame, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
            return cv2.inpaint(frame, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
        def update_progress(self, percent):
            self.progress["value"] = percent
            self.status_label.config(text=f"Processing: {percent}%")
            self.update_idletasks()
    
        def update_status(self, message):
            self.status_label.config(text=message)
            self.update_idletasks()
    
    if __name__ == "__main__":
        app = VideoEditorApp()
        app.mainloop()
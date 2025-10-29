import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import sys, os, json
import shutil
from pathlib import Path
import cv2
import numpy as np
import uuid, datetime

# Ensure project root on sys.path for backend imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.models import Image as ImageModel, Defect as DefectModel, Report as ReportModel
from automated_defect_detection.database_manager import save_image, save_defects, save_report, init_db

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class ImageUploadApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Upload & Analysis")
        self.geometry("1000x700")
        self.configure(fg_color="#f0f2f5")  # Light grey background
        self.image_path = None
        self.model = None
        self.class_names = ["negative", "positive"]
        self.img_size = 224  # must match training export

        # Configure the main grid
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_main_content()
        self._load_classifier()
        # Ensure upload dir exists inside project
        try:
            self.upload_dir = Path(PROJECT_ROOT) / "automated_defect_detection" / "uploaded_images"
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.upload_dir = Path.cwd()
        try:
            init_db()
        except Exception:
            pass
    
    def go_back(self):
        # Example: destroy current window
        self.destroy()

    def _create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=0)  # Back arrow column
        header_frame.grid_columnconfigure(1, weight=1)  # Title column

        back_button = ctk.CTkButton(header_frame, text="←", width=40, height=40,
                        fg_color="transparent", text_color="#333",
                        font=("Roboto", 24, "bold"), hover_color="#f0f2f5",
                        command=self.go_back)
        back_button.grid(row=0, column=0, padx=(10, 5), pady=10)

        title_label = ctk.CTkLabel(header_frame, text="Image Upload & Analysis",
                                   font=("Roboto", 20, "bold"), text_color="#333")
        title_label.grid(row=0, column=1, sticky="w", padx=10, pady=10)

        subtitle_label = ctk.CTkLabel(header_frame, text="Upload images for automated defect detection",
                                      font=("Roboto", 12), text_color="#666")
        subtitle_label.grid(row=0, column=1, sticky="w", padx=(230, 0))

    def _create_main_content(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)  # Upload section
        main_frame.grid_columnconfigure(1, weight=1)  # Preview section

        self._create_upload_section(main_frame)
        self._create_preview_section(main_frame)

    def _create_upload_section(self, parent_frame):
        upload_frame = ctk.CTkFrame(parent_frame, fg_color="white", corner_radius=10)
        upload_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        upload_frame.grid_rowconfigure(0, weight=1)
        upload_frame.grid_rowconfigure(1, weight=0)
        upload_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(upload_frame, text="Upload Image", font=("Roboto", 16, "bold"),
                     text_color="#333").pack(pady=(20, 5), anchor="w", padx=20)
        ctk.CTkLabel(upload_frame, text="Select or drag and drop an image for defect analysis",
                     font=("Roboto", 12), text_color="#666").pack(pady=(0, 20), anchor="w", padx=20)

        # Upload box
# Old, incorrect code:
# upload_box = ctk.CTkFrame(upload_frame, fg_color="#f7f7f7", corner_radius=10,
#                           border_width=2, border_color="#d4d4d4", border_style="dashed")

# Corrected code:
        upload_box = ctk.CTkFrame(upload_frame, fg_color="#f7f7f7", corner_radius=10,
                          border_width=2, border_color="#d4d4d4")
        upload_box.pack(pady=(0, 20), padx=20, fill="both", expand=True)

        ctk.CTkLabel(upload_box, text="⬆️", font=("Segoe UI Emoji", 48), text_color="#999").pack(pady=(40, 5))
        ctk.CTkLabel(upload_box, text="Drop your image here", font=("Roboto", 14), text_color="#666").pack()
        ctk.CTkLabel(upload_box, text="or click to browse files", font=("Roboto", 12), text_color="#999").pack(pady=(0, 10))

        # This button is for a single click upload functionality
        browse_button = ctk.CTkButton(upload_box, text="Click to Browse", command=self.browse_files,
                                      fg_color="transparent", text_color="#3b82f6", hover_color="#e6f0ff",
                                      font=("Roboto", 12, "bold"))
        browse_button.pack(pady=(0, 40))

        # Preprocess button
        preprocess_button = ctk.CTkButton(upload_frame, text="🔬 Preprocess & Detect Defects", width=250, height=40,
                                          font=("Roboto", 14, "bold"), corner_radius=10)
        preprocess_button.pack(pady=(0, 20))

        # Classification UI additions
        self.result_label = ctk.CTkLabel(upload_frame, text="", font=("Roboto", 13), text_color="#333")
        self.result_label.pack(pady=(0, 10))

        classify_button = ctk.CTkButton(upload_frame, text="Classify Image", width=250, height=40,
                                        font=("Roboto", 14, "bold"), corner_radius=10, command=self.classify_image)
        classify_button.pack(pady=(0, 20))

    def _create_preview_section(self, parent_frame):
        preview_frame = ctk.CTkFrame(parent_frame, fg_color="white", corner_radius=10)
        preview_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        preview_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(preview_frame, text="Image Preview", font=("Roboto", 16, "bold"),
                     text_color="#333").pack(pady=(20, 5), anchor="w", padx=20)

        self.preview_label = ctk.CTkLabel(preview_frame, text="No image selected", font=("Roboto", 14),
                                          text_color="#999", image=None)
        self.preview_label.pack(expand=True, fill="both", pady=20, padx=20)

    def browse_files(self):
        """Opens a file dialog to select an image and displays it in the preview section."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.image_path = file_path
            # Copy chosen file into app's uploaded_images folder
            try:
                src = Path(self.image_path)
                dest = self.upload_dir / src.name
                if not dest.exists() or src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                # Use copied path going forward
                self.image_path = str(dest)
            except Exception as e:
                try:
                    self.result_label.configure(text=f"Copy failed: {e}", text_color="#ef4444")
                except Exception:
                    pass
            self.update_preview()
            try:
                self.result_label.configure(text="")
            except Exception:
                pass

    def update_preview(self):
        if self.image_path:
            # Open the image and resize it to fit the preview frame
            pil_image = Image.open(self.image_path)
            pil_image.thumbnail((500, 500))  # Resize to fit
            
            # Convert to PhotoImage for Tkinter
            tk_image = ctk.CTkImage(light_image=pil_image, size=(pil_image.width, pil_image.height))
            
            # Update the preview label with the new image
            self.preview_label.configure(image=tk_image, text="")
            self.preview_label.image = tk_image # Keep a reference to avoid garbage collection

    def _load_classifier(self):
        try:
            project_root = Path(__file__).resolve().parents[1]
            clf_dir = project_root / "automated_defect_detection" / "trained" / "classifier"
            onnx_path = clf_dir / "best.onnx"
            label_map_path = clf_dir / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, "r", encoding="utf-8") as f:
                    label_map = json.load(f)
                if isinstance(label_map, dict):
                    self.class_names = [label_map[str(i)] for i in range(len(label_map)) if str(i) in label_map]
                elif isinstance(label_map, list):
                    self.class_names = label_map
            if not onnx_path.exists():
                return
            self.model = cv2.dnn.readNetFromONNX(str(onnx_path))
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            self.model = None

    def _preprocess_for_onnx(self, img_path: str):
        bgr = cv2.imread(img_path)
        if bgr is None:
            raise RuntimeError("Failed to read image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        chw = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(chw, axis=0)  # NCHW
        return blob

    def classify_image(self):
        if not self.image_path:
            from tkinter import messagebox
            messagebox.showinfo("Info", "Please choose an image first.")
            return
        # Ensure selected image is inside our upload_dir
        try:
            src = Path(self.image_path)
            if self.upload_dir not in src.parents:
                dest = self.upload_dir / src.name
                if not dest.exists() or src.resolve() != dest.resolve():
                    shutil.copy2(src, dest)
                self.image_path = str(dest)
        except Exception:
            pass
        if self.model is None:
            self.result_label.configure(text="Model not loaded. Train the model to create best.onnx.", text_color="#ef4444")
            return
        try:
            inp = self._preprocess_for_onnx(self.image_path)
            self.model.setInput(inp)
            logits = self.model.forward()
            logits = np.array(logits).reshape(1, -1)
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            idx = int(np.argmax(probs, axis=1)[0])
            conf = float(probs[0, idx])
            label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
            self.result_label.configure(text=f"Prediction: {label} ({conf:.2%})", text_color="#10b981")
            # Persist result to DB as image/report with optional defect row
            self._save_classification_to_db(label, conf)
        except Exception as e:
            self.result_label.configure(text=f"Inference error: {e}", text_color="#ef4444")

    def _save_classification_to_db(self, label: str, confidence: float):
        try:
            os.makedirs("reports", exist_ok=True)
            filename = os.path.basename(self.image_path)
            # Image metadata
            img = ImageModel(user_id=None, filename=filename, original_path=self.image_path, processed_path=None, status="classified")
            try:
                img.uploadDate = datetime.datetime.now()
            except Exception:
                pass
            save_image(img)

            defects = []
            if label.lower() == "positive":
                d = DefectModel(image_id=img.imageID, defect_type="positive", bounding_box=[], confidence=confidence)
                defects.append(d)
                save_defects(defects)

            # Report
            report_id = str(uuid.uuid4())
            report_path = os.path.join("reports", f"{report_id}.json")
            rpt = ReportModel(report_id, img.imageID, defect_count=len(defects), report_path=report_path)
            try:
                rpt.reportDate = datetime.datetime.now()
            except Exception:
                pass
            save_report(rpt)

            payload = {
                "reportID": rpt.reportID,
                "image": {
                    "imageID": img.imageID,
                    "filename": img.filename,
                    "originalPath": img.originalPath,
                    "processedPath": img.processedPath,
                },
                "reportDate": rpt.reportDate if isinstance(rpt.reportDate, str) else rpt.reportDate.isoformat(),
                "defectCount": rpt.defectCount,
                "prediction": label,
                "confidence": confidence,
                "defects": [
                    {
                        "defectID": d.defectID,
                        "type": d.type,
                        "confidence": d.confidence,
                        "boundingBox": d.boundingBox,
                    }
                    for d in defects
                ],
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            try:
                self.result_label.configure(text=f"Saved prediction. DB/report save warning: {e}", text_color="#f39c12")
            except Exception:
                pass

if __name__ == "__main__":
    app = ImageUploadApp()
    app.mainloop()

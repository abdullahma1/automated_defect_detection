import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import subprocess
import sys

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

        # Configure the main grid
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main content
        self.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_main_content()
    
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
            self.update_preview()

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

if __name__ == "__main__":
    app = ImageUploadApp()
    app.mainloop()
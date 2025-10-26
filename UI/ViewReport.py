import customtkinter as ctk
from PIL import Image

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class DefectReportApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Defect Analysis Report")
        self.geometry("1400x900")
        self.configure(fg_color="#f0f2f5") # Light grey background

        # Configure the main window grid
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Main content area
        self.grid_columnconfigure(0, weight=1)
        
        # Dummy data for the report
        self.report_data = {
            "id": "RPT-1703924800000",
            "image_file": "circuit_board_001.jpg",
            "analysis_date": "1/8/2024, 7:30:00 PM",
            "processing_time": "2.3s",
            "total_defects": 2,
            "defects": [
                {
                    "id": 1,
                    "type": "Crack",
                    "confidence": "94%",
                    "position": "(120, 80)",
                    "dimensions": "45x30px",
                    "area": "1350px²",
                    "severity": "Critical"
                },
                {
                    "id": 2,
                    "type": "Scratch",
                    "confidence": "87%",
                    "position": "(280, 150)",
                    "dimensions": "60x15px",
                    "area": "900px²",
                    "severity": "High"
                }
            ]
        }
        
        self._create_header()
        self._create_analysis_summary()
        self._create_main_content()
        self._create_analysis_statistics()

    def go_back(self):
        # Example: destroy current window
        self.destroy()

    def _create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=0) # Back arrow
        header_frame.grid_columnconfigure(1, weight=1) # Report ID
        header_frame.grid_columnconfigure(2, weight=0) # Export button
        header_frame.grid_columnconfigure(3, weight=0) # Download button
        
        
        back_button = ctk.CTkButton(header_frame, text="←", width=40, height=40,
                        fg_color="transparent", text_color="#333",
                        font=("Roboto", 24, "bold"), hover_color="#f0f2f5",
                        command=self.go_back)
        back_button.grid(row=0, column=0, padx=(10, 5), pady=10)

        title_label = ctk.CTkLabel(header_frame, text=f"Defect Analysis Report", font=("Roboto", 20, "bold"), text_color="#333")
        title_label.grid(row=0, column=1, sticky="w", padx=10, pady=10)
        subtitle_label = ctk.CTkLabel(header_frame, text=f"Report ID: {self.report_data['id']}", font=("Roboto", 12), text_color="#666")
        subtitle_label.grid(row=0, column=1, sticky="w", padx=(190, 0))

        export_csv_button = ctk.CTkButton(header_frame, text="Export CSV", width=120, height=40)
        export_csv_button.grid(row=0, column=2, padx=(0, 5), pady=10)
        
        download_pdf_button = ctk.CTkButton(header_frame, text="Download PDF", width=120, height=40, fg_color="#3b82f6")
        download_pdf_button.grid(row=0, column=3, padx=(0, 20), pady=10)

    def _create_analysis_summary(self):
        summary_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=10)
        summary_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(10, 20))
        summary_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Helper function to create a summary card
        def create_card(parent, col, title, value, icon, icon_color_key, value_color="#333"):
            card_frame = ctk.CTkFrame(parent, fg_color="white")
            card_frame.grid(row=0, column=col, sticky="nsew", padx=10, pady=10)
            card_frame.grid_columnconfigure((0, 1), weight=1)

            ctk.CTkLabel(card_frame, text=icon, font=("Segoe UI Emoji", 30), text_color="#3b82f6").grid(row=0, column=0, sticky="w", padx=(0, 10))
            ctk.CTkLabel(card_frame, text=title, font=("Roboto", 14), text_color="#666").grid(row=0, column=1, sticky="w")
            ctk.CTkLabel(card_frame, text=value, font=("Roboto", 16, "bold"), text_color=value_color).grid(row=1, column=1, sticky="w")
        
        create_card(summary_frame, 0, "Image File", self.report_data["image_file"], "📄", "blue")
        create_card(summary_frame, 1, "Analysis Date", self.report_data["analysis_date"], "📅", "blue")
        create_card(summary_frame, 2, "Processing Time", self.report_data["processing_time"], "⏱️", "blue")
        create_card(summary_frame, 3, f"{self.report_data['total_defects']} Defects Found", f"{self.report_data['total_defects']} defects", "❗", "red", value_color="#e74c3c")

    def _create_main_content(self):
        main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_content_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=0)
        main_content_frame.grid_columnconfigure(0, weight=1)
        main_content_frame.grid_columnconfigure(1, weight=1)
        
        # --- Processed Image Section ---
        image_frame = ctk.CTkFrame(main_content_frame, fg_color="white", corner_radius=10)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        ctk.CTkLabel(image_frame, text="Processed Image", font=("Roboto", 16, "bold")).pack(pady=(20, 5), padx=20, anchor="w")
        ctk.CTkLabel(image_frame, text="Red bounding boxes indicate detected defects", font=("Roboto", 10), text_color="#666").pack(padx=20, anchor="w")
        
        try:
            # Note: You need a local image file named 'circuit_board.png' for this to work
            pil_image = Image.open("circuit_board.png")
            ctk_image = ctk.CTkImage(light_image=pil_image, size=(500, 300))
            image_label = ctk.CTkLabel(image_frame, image=ctk_image, text="")
            image_label.pack(pady=20, padx=20)
        except FileNotFoundError:
            image_placeholder = ctk.CTkLabel(image_frame, text="[Image Placeholder]", font=("Roboto", 20), text_color="#999")
            image_placeholder.pack(expand=True, fill="both")
            
        # --- Defect Details Section ---
        details_frame = ctk.CTkFrame(main_content_frame, fg_color="white", corner_radius=10)
        details_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)
        ctk.CTkLabel(details_frame, text="Defect Details", font=("Roboto", 16, "bold")).pack(pady=(20, 5), padx=20, anchor="w")
        ctk.CTkLabel(details_frame, text="Complete list of detected defects with confidence scores", font=("Roboto", 10), text_color="#666").pack(padx=20, anchor="w")
        
        for defect in self.report_data['defects']:
            self._create_defect_card(details_frame, defect)

    def _create_defect_card(self, parent, defect_data):
        card_frame = ctk.CTkFrame(parent, fg_color="#f7f7f7", corner_radius=10)
        card_frame.pack(fill="x", padx=20, pady=10)
        card_frame.grid_columnconfigure(0, weight=1)
        card_frame.grid_columnconfigure(1, weight=1)

        severity_color = {"Critical": "#e74c3c", "High": "#f39c12", "Medium": "#f1c40f", "Low": "#2ecc71"}.get(defect_data['severity'], "#95a5a6")

        # Row 1
        ctk.CTkLabel(card_frame, text=f"#{defect_data['id']}", font=("Roboto", 12, "bold")).grid(row=0, column=0, sticky="w", padx=15, pady=(15, 0))
        ctk.CTkLabel(card_frame, text=f"{defect_data['type']}", font=("Roboto", 16, "bold")).grid(row=0, column=0, sticky="w", padx=(40, 0), pady=(15, 0))
        ctk.CTkLabel(card_frame, text=f" {defect_data['severity']} ", fg_color=severity_color, text_color="white", corner_radius=5).grid(row=0, column=1, sticky="e", padx=(0, 15), pady=(15, 0))
        
        # Row 2
        ctk.CTkLabel(card_frame, text=f"Confidence: {defect_data['confidence']}", font=("Roboto", 12), text_color="#666").grid(row=1, column=0, sticky="w", padx=15)
        ctk.CTkLabel(card_frame, text=f"Position: {defect_data['position']}", font=("Roboto", 12), text_color="#666").grid(row=1, column=1, sticky="e", padx=15)
        
        # Row 3
        ctk.CTkLabel(card_frame, text=f"Dimensions: {defect_data['dimensions']}", font=("Roboto", 12), text_color="#666").grid(row=2, column=0, sticky="w", padx=15, pady=(0, 15))
        ctk.CTkLabel(card_frame, text=f"Area: {defect_data['area']}", font=("Roboto", 12), text_color="#666").grid(row=2, column=1, sticky="e", padx=15, pady=(0, 15))


    def _create_analysis_statistics(self):
        stats_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=10)
        stats_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(10, 20))
        stats_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        ctk.CTkLabel(stats_frame, text="Analysis Statistics", font=("Roboto", 16, "bold")).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10), columnspan=7)
        
        stats = {
            "Total Defects": self.report_data['total_defects'],
            "Critical": 1,
            "High": 1,
            "Medium": 0,
            "Low": 0,
            "Avg Confidence": "91%"
        }
        
        for i, (label, value) in enumerate(stats.items()):
            label_text = ctk.CTkLabel(stats_frame, text=label, font=("Roboto", 12, "bold"), text_color="#666")
            label_text.grid(row=1, column=i, pady=(0, 5), padx=5)
            value_text = ctk.CTkLabel(stats_frame, text=str(value), font=("Roboto", 24), text_color="#333")
            value_text.grid(row=2, column=i, pady=(0, 20), padx=5)

if __name__ == "__main__":
    app = DefectReportApp()
    app.mainloop()
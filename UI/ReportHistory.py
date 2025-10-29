import customtkinter as ctk
import subprocess
import sys, os, csv
from tkinter import filedialog

# Ensure project root on sys.path for backend imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.database_manager import fetch_recent_reports

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class AnalysisHistoryApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Analysis History")
        self.geometry("1200x800")
        self.configure(fg_color="#f0f2f5") # Light grey background

        # Configure grid to be responsive
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=0) # Search bar
        self.grid_rowconfigure(2, weight=0) # Summary cards
        self.grid_rowconfigure(3, weight=1) # Reports list
        self.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_search_bar()
        self._create_summary_cards()
        self.reports = []
        self._create_reports_list()
    
    def go_back(self):
        # Example: destroy current window
        self.destroy()

    def _create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        header_frame.grid_columnconfigure(0, weight=0) # Back arrow column
        header_frame.grid_columnconfigure(1, weight=1) # Title column

        back_button = ctk.CTkButton(header_frame, text="←", width=40, height=40,
                                    fg_color="transparent", text_color="#333",
                                    font=("Roboto", 24, "bold"), hover_color="#f0f2f5",command=self.go_back)
        
        
        back_button.grid(row=0, column=0, padx=(10, 5), pady=10)

        title_label = ctk.CTkLabel(header_frame, text="Analysis History",
                                   font=("Roboto", 20, "bold"), text_color="#333")
        title_label.grid(row=0, column=1, sticky="w", padx=10, pady=10)

        subtitle_label = ctk.CTkLabel(header_frame, text="View and manage previous defect analysis reports",
                                      font=("Roboto", 12), text_color="#666")
        subtitle_label.grid(row=0, column=1, sticky="w", padx=(180, 0)) # Position subtitle relative to title

    def _create_search_bar(self):
        search_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=10)
        search_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        search_frame.grid_columnconfigure(0, weight=1) # Search bar column
        search_frame.grid_columnconfigure(1, weight=0) # Date range column

        search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search by filename or report ID...",
                                    width=700, height=40)
        search_entry.grid(row=0, column=0, sticky="ew", padx=(20, 10), pady=10)

        date_range_button = ctk.CTkButton(search_frame, text="📅 Date Range", width=150, height=40)
        date_range_button.grid(row=0, column=1, sticky="e", padx=(0, 20), pady=10)

    def _create_summary_cards(self):
        summary_frame = ctk.CTkFrame(self, fg_color="transparent")
        summary_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        summary_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Helper function to create a summary card
        def create_card(parent, row, col, title, value, icon, icon_color_key, value_color="#333"):
            card_frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=10)
            card_frame.grid(row=row, column=col, sticky="nsew", padx=10, pady=10)
            card_frame.grid_columnconfigure((0, 1), weight=1)

            ctk.CTkLabel(card_frame, text=title, font=("Roboto", 14), text_color="#666").grid(row=0, column=0, sticky="nw", padx=20, pady=(15, 0))
            ctk.CTkLabel(card_frame, text=value, font=("Roboto", 28, "bold"), text_color=value_color).grid(row=1, column=0, sticky="nw", padx=20, pady=(5, 15))

            icon_color = {"blue": "#3b82f6", "green": "#2ecc71", "red": "#e74c3c", "gray": "#95a5a6"}.get(icon_color_key, "#888")
            icon_label = ctk.CTkLabel(card_frame, text=icon, font=("Segoe UI Emoji", 36), text_color=icon_color)
            icon_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 20))

        create_card(summary_frame, 0, 0, "Total Reports", "5", "📄", "blue")
        create_card(summary_frame, 0, 1, "Images Analyzed", "5", "🖼️", "green")
        create_card(summary_frame, 0, 2, "Defects Found", "11", "❗", "red")
        create_card(summary_frame, 0, 3, "Avg Analysis Time", "2.3s", "⏱️", "gray")

    def _create_reports_list(self):
        list_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=10)
        list_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            list_frame, 
            text="🔄 Refresh", 
            width=100,
            command=self.refresh_reports,
            fg_color="transparent",
            text_color="#3b82f6"
        )
        refresh_btn.pack(pady=(10,0), anchor="e", padx=20)

        title_label = ctk.CTkLabel(list_frame, text="Recent Analysis Reports", font=("Roboto", 20, "bold"), text_color="#333")
        title_label.pack(pady=(20, 5), anchor="w", padx=20)
        try:
            self.reports = fetch_recent_reports(limit=50)
        except Exception:
            self.reports = []
        subtitle_label = ctk.CTkLabel(list_frame, text=f"{len(self.reports)} reports", font=("Roboto", 12), text_color="#666")
        subtitle_label.pack(pady=(0, 10), anchor="w", padx=20)
        ctk.CTkButton(list_frame, text="Export CSV", width=120, height=32, command=self.export_csv).pack(pady=(0, 10), anchor="e", padx=20)

        for row in self.reports:
            desc = row.get("filename") or "(unknown)"
            report_id = row.get("reportID")
            timestamp = str(row.get("reportDate"))
            analysis_time = "-"
            defect_count = row.get("defectCount", 0) or 0
            status_text = f"{defect_count} defects"
            status_color = "#e74c3c" if defect_count > 0 else "#2ecc71"
            self._create_report_item(list_frame, desc, report_id, timestamp, analysis_time, status_text, status_color)

    def _create_report_item(self, parent_frame, filename, report_id, date_time, analysis_time, status_text, status_color):
        item_frame = ctk.CTkFrame(parent_frame, fg_color="#f7f7f7", corner_radius=10, height=80)
        item_frame.pack(fill="x", pady=5, padx=20)
        item_frame.grid_columnconfigure(0, weight=0) # Icon
        item_frame.grid_columnconfigure(1, weight=1) # Text details
        item_frame.grid_columnconfigure(2, weight=0) # Status badge
        item_frame.grid_columnconfigure(3, weight=0) # View button
        item_frame.grid_columnconfigure(4, weight=0) # Download button

        # Icon placeholder
        icon_label = ctk.CTkLabel(item_frame, text="📄", font=("Segoe UI Emoji", 24), text_color="#3b82f6")
        icon_label.grid(row=0, column=0, rowspan=2, padx=(20, 10), pady=10, sticky="nsew")

        # Filename and details
        filename_label = ctk.CTkLabel(item_frame, text=filename, font=("Roboto", 14, "bold"), anchor="w", text_color="#333")
        filename_label.grid(row=0, column=1, sticky="w", pady=(10, 0))
        details_label = ctk.CTkLabel(item_frame, text=f"{report_id}  |  {date_time}  |  Analysis: {analysis_time}",
                                     font=("Roboto", 10), anchor="w", text_color="#666")
        details_label.grid(row=1, column=1, sticky="w", pady=(0, 10))

        # Status Badge
        status_label = ctk.CTkLabel(item_frame, text=f" {status_text} ", fg_color=status_color, text_color="white", corner_radius=5)
        status_label.grid(row=0, column=2, sticky="e", padx=(10, 5), pady=(10, 0))
        status_completed_label = ctk.CTkLabel(item_frame, text="completed", text_color="#666", font=("Roboto", 10))
        status_completed_label.grid(row=1, column=2, sticky="e", padx=(10, 5))

        # Action Buttons
        view_button = ctk.CTkButton(item_frame, text="View", width=70, height=30, font=("Roboto", 12),
                                    command=lambda rid=report_id: self.open_view(rid))
        view_button.grid(row=0, column=3, rowspan=2, padx=(5, 5), pady=10)
        download_button = ctk.CTkButton(item_frame, text="⬇️", width=40, height=30,
                                        fg_color="transparent", text_color="#3b82f6", font=("Segoe UI Emoji", 14),
                                        hover_color="#e6f0ff")
        download_button.grid(row=0, column=4, rowspan=2, padx=(0, 20), pady=10)

    def refresh_reports(self):
        """Refresh the reports list with latest data"""
        try:
            self.reports = fetch_recent_reports(limit=50)
            for widget in self.winfo_children():
                widget.destroy()
            self._create_reports_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh reports: {e}")

    def open_view(self, report_id=None):
        """Open the report viewer with specific report"""
        try:
            cmd = [sys.executable, "UI/ViewReport.py"]
            if report_id:
                cmd.append(f"--report={report_id}")
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")

    def export_csv(self):
        try:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if not path:
                return
            if not getattr(self, 'reports', None):
                self.reports = fetch_recent_reports(limit=50)
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["reportID", "imageID", "filename", "reportDate", "defectCount", "reportPath", "userID"])
                for r in self.reports:
                    writer.writerow([
                        r.get("reportID"), r.get("imageID"), r.get("filename"), r.get("reportDate"), r.get("defectCount"), r.get("reportPath"), r.get("userID")
                    ])
        except Exception:
            pass

if __name__ == "__main__":
    app = AnalysisHistoryApp()
    app.mainloop()
    

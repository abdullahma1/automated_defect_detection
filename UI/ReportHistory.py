import customtkinter as ctk
import subprocess
import sys, os, csv, json, shutil
from datetime import datetime
from tkinter import filedialog, messagebox

# Ensure project root on sys.path for backend imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.database_manager import count_reports, count_images,count_defects,  fetch_recent_reports, fetch_report_byusername, fetch_report_details

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
        search_frame.grid_columnconfigure(1, weight=0) # Filter button column

        self.search_entry = ctk.CTkEntry(search_frame, placeholder_text="Search by username or filename...",
                                    width=700, height=40)
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(20, 10), pady=10)

        filter_button = ctk.CTkButton(
            search_frame,
            text="🔍 Filter",
            width=150,
            height=40,
            command=self.filter_reports
        )
        filter_button.grid(row=0, column=1, sticky="e", padx=(0, 20), pady=10)

        # ---------- ADD THIS: persistent result label ----------
        # Create result_label once so filter_reports can safely call .configure()
        self.result_label = ctk.CTkLabel(search_frame, text="", font=("Roboto", 12), text_color="#666")
        self.result_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=(20,0), pady=(0,5))

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

        total_reports = count_reports()  # Fetch total reports for summary
        images_analyzed = count_images() # Placeholder, replace with actual count if available
        defects_found = count_defects() # Placeholder, replace with actual count if available
        create_card(summary_frame, 0, 0, "Total Reports", total_reports, "📄", "blue")
        create_card(summary_frame, 0, 1, "Images Analyzed", images_analyzed, "🖼️", "green")
        create_card(summary_frame, 0, 2, "Defects Found", defects_found, "❗", "red")
        create_card(summary_frame, 0, 3, "Avg Analysis Time", "2.3s", "⏱️", "gray")

    def filter_reports(self):
        search_term = self.search_entry.get().strip()

        try:
            # Filtered data lo
            self.reports = fetch_report_byusername(search_term)

            # purana list frame hatao
            if hasattr(self, "list_frame") and self.list_frame.winfo_exists():
                try:
                    self.list_frame.destroy()
                except Exception:
                    pass

            # naya list frame banao
            self._create_reports_list(search_term)  # ye khud _build_reports_ui() call karega

            # feedback show karo
            count = len(self.reports)
            if count == 0:
                self.result_label.configure(text=f"No reports found matching '{search_term}'", text_color="#666")
            else:
                self.result_label.configure(text=f"No record Found", text_color="#ab0d0d")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to filter reports: {e}")



    def refresh_reports(self):
        """Refresh the reports list with latest data"""
        try:
            self.reports = fetch_recent_reports(limit=50)
            # Rebuild the scrollable list area only
            if hasattr(self, "list_frame") and self.list_frame.winfo_exists():
                try:
                    self.list_frame.destroy()
                except Exception:
                    pass
            self._create_reports_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh reports: {e}")

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


    def _create_reports_list(self , username_entry=""):
        # Scrollable container for long report lists
        self.list_frame = ctk.CTkScrollableFrame(self, fg_color="white", corner_radius=10, height=480)
        self.list_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.list_frame.grid_columnconfigure(0, weight=1)

        # Refresh button
        refresh_btn = ctk.CTkButton(
            self.list_frame, 
            text="🔄 Refresh", 
            width=100,
            command=self.refresh_reports,
            fg_color="transparent",
            text_color="#3b82f6"
        )
        refresh_btn.pack(pady=(10,0), anchor="e", padx=20)

        title_label = ctk.CTkLabel(self.list_frame, text="Recent Analysis Reports", font=("Roboto", 20, "bold"), text_color="#333")
        title_label.pack(pady=(20, 5), anchor="w", padx=20)

        # agar abhi koi data load nahi hua to latest 50 reports le lo
        if not getattr(self, "reports", None) and (username_entry == ""):
            try:
                self.reports = fetch_recent_reports(limit=50)
            except Exception:
                self.reports = []

        subtitle_label = ctk.CTkLabel(self.list_frame, text=f"{len(self.reports)} reports", font=("Roboto", 12), text_color="#666")
        subtitle_label.pack(pady=(0, 10), anchor="w", padx=20)
        ctk.CTkButton(self.list_frame, text="Export CSV", width=120, height=32, command=self.export_csv).pack(pady=(0, 10), anchor="e", padx=20)

        # ab yahan build function call kar do 👇
        self._build_reports_ui()


    def _build_reports_ui(self):
        for row in self.reports:
            desc = row.get("username") or row.get("filename") or "(unknown)"
            report_id = row.get("reportID")
            timestamp = str(row.get("reportDate"))
            defect_count = row.get("defectCount", 0) or 0
            status_text = f"{defect_count} defects"
            status_color = "#e74c3c" if defect_count > 0 else "#2ecc71"
            self._create_report_item(self.list_frame, desc, report_id, timestamp, "-", status_text, status_color, row)


    def _create_report_item(self, parent_frame, filename, report_id, date_time, analysis_time, status_text, status_color, row):
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

        # Main label - show username or filename
        display_text = filename  # filename parameter now contains either username or filename
        if row.get("username"):  # If we're showing username, also show the filename below
            filename_text = row.get("filename", "(unknown)")
            details_text = f"File: {filename_text}"
        else:
            details_text = "File: " + filename  # filename parameter contains the filename in this case
            
        main_label = ctk.CTkLabel(item_frame, text=display_text, font=("Roboto", 14, "bold"), anchor="w", text_color="#333")
        main_label.grid(row=0, column=1, sticky="w", pady=(10, 0))
        
        # Details including report ID and date
        details_label = ctk.CTkLabel(item_frame, text=f"{details_text}  |  {report_id}  |  {date_time}  |  Analysis: {analysis_time}",
                                    font=("Roboto", 12), anchor="w", text_color="#666")
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
                                        hover_color="#e6f0ff",
                                        command=lambda rid=report_id: self.download_report(rid))
        download_button.grid(row=0, column=4, rowspan=2, padx=(0, 20), pady=10)

    def download_report(self, report_id):
        """Download a single report: prefer stored report JSON file; otherwise serialize DB details."""
        try:
            details = fetch_report_details(report_id)
        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch report details: {e}")
            return

        # Determine default filename
        default_name = f"{report_id or 'report'}.json"
        # Try to use file dialog to choose destination
        dest = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON Files", "*.json")],
                                            initialfile=default_name)
        if not dest:
            return

        # If a reportPath exists and file is present, copy it; else dump 'details' as JSON
        try:
            report_path = None
            if isinstance(details, dict):
                report_path = details.get("reportPath")
            if report_path and os.path.isfile(report_path):
                shutil.copy2(report_path, dest)
            else:
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(details or {}, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Report saved to:\n{dest}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")

    def open_view(self, report_id=None):
        """Open the report viewer with specific report"""
        try:
            cmd = [sys.executable, "UI/ViewReport.py"]
            if report_id:
                cmd.append(f"--report={report_id}")
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")

class DateRangeDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Date Range")
        self.geometry("400x300")
        
        self.result = None  # Will store (start_date, end_date) if OK clicked
        
        # Center on parent
        self.transient(parent)
        self.grab_set()
        
        x = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        y = parent.winfo_y() + (parent.winfo_height() - 300) // 2
        self.geometry(f"+{x}+{y}")
        
        # Date picker frames
        start_frame = ctk.CTkFrame(self)
        start_frame.pack(padx=20, pady=(20,10), fill="x")
        
        end_frame = ctk.CTkFrame(self)
        end_frame.pack(padx=20, pady=10, fill="x")
        
        # Labels
        ctk.CTkLabel(start_frame, text="Start Date:").pack(side="left", padx=5)
        ctk.CTkLabel(end_frame, text="End Date:").pack(side="left", padx=5)
        
        # Date entry boxes (YYYY-MM-DD)
        self.start_entry = ctk.CTkEntry(start_frame, placeholder_text="YYYY-MM-DD")
        self.start_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        self.end_entry = ctk.CTkEntry(end_frame, placeholder_text="YYYY-MM-DD")
        self.end_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Buttons
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20, fill="x")
        
        ctk.CTkButton(
            button_frame,
            text="Apply Filter",
            command=self._on_ok
        ).pack(side="left", padx=10, expand=True)
        
        ctk.CTkButton(
            button_frame,
            text="Clear Filter",
            command=self._on_clear,
            fg_color="transparent",
            text_color="#3b82f6"
        ).pack(side="left", padx=10, expand=True)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            fg_color="#f3f4f6",
            text_color="#374151"
        ).pack(side="left", padx=10, expand=True)
        
        self.wait_window()
    
    def _on_ok(self):
        """Validate and store the date range"""
        try:
            start = self.start_entry.get().strip()
            end = self.end_entry.get().strip()
            
            # Basic validation
            if start and not self._is_valid_date(start):
                messagebox.showerror("Error", "Invalid start date format. Use YYYY-MM-DD")
                return
            if end and not self._is_valid_date(end):
                messagebox.showerror("Error", "Invalid end date format. Use YYYY-MM-DD")
                return
                
            # Ensure end date is not before start date
            if start and end:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
                end_dt = datetime.strptime(end, "%Y-%m-%d")
                if end_dt < start_dt:
                    messagebox.showerror("Error", "End date cannot be before start date")
                    return
                
            self.result = (start if start else None, end if end else None)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid date format: {e}")
    
    def _on_clear(self):
        """Clear the filter"""
        self.result = (None, None)
        self.destroy()
    
    def _on_cancel(self):
        """Cancel without changes"""
        self.destroy()
    
    def _is_valid_date(self, date_str):
        """Simple date format validation"""
        try:
            # Check basic format
            parts = date_str.split("-")
            if len(parts) != 3:
                return False
            
            year, month, day = map(int, parts)
            
            # Basic range checks
            return (
                len(str(year)) == 4 and
                1 <= month <= 12 and
                1 <= day <= 31
            )
        except:
            return False
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
                                        hover_color="#e6f0ff",
                                        command=lambda rid=report_id: self.download_report(rid))
        download_button.grid(row=0, column=4, rowspan=2, padx=(0, 20), pady=10)

    def download_report(self, report_id):
        """Download a single report: prefer stored report JSON file; otherwise serialize DB details."""
        try:
            details = fetch_report_details(report_id)
        except Exception as e:
            messagebox.showerror("Error", f"Could not fetch report details: {e}")
            return

        # Determine default filename
        default_name = f"{report_id or 'report'}.json"
        # Try to use file dialog to choose destination
        dest = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON Files", "*.json")],
                                            initialfile=default_name)
        if not dest:
            return

        # If a reportPath exists and file is present, copy it; else dump 'details' as JSON
        try:
            report_path = None
            if isinstance(details, dict):
                report_path = details.get("reportPath")
            if report_path and os.path.isfile(report_path):
                shutil.copy2(report_path, dest)
            else:
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(details or {}, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Saved", f"Report saved to:\n{dest}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {e}")

    def open_view(self, report_id=None):
        """Open the report viewer with specific report"""
        try:
            cmd = [sys.executable, "UI/ViewReport.py"]
            if report_id:
                cmd.append(f"--report={report_id}")
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")

if __name__ == "__main__":
    app = AnalysisHistoryApp()
    app.mainloop()
    app.mainloop()
    

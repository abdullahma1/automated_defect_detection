import customtkinter as ctk
import subprocess
import sys

from login import LoginApp

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class DashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DefectScan Pro - Dashboard")
        self.geometry("1200x800")
        self.configure(fg_color="#f0f2f5")

        # Layout
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_main_content()

    def log_out(self):
    # Example: clear session, close window, or redirect to login
    # If you just want to close the whole app:
        self.show_login_page()
        

    def show_login_page(self):
        try:
            #subprocess.Popen([sys.executable, "login.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
            login = LoginApp()
            login.withdraw()
            app.destroy()

            app.destroy()
        except Exception:
                # Fallback: try without creationflags on non-Windows platforms
            subprocess.Popen([sys.executable, "UI/login.py"])
            app.destroy()
            app.withdraw()

            


    def _create_header(self):
        header_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        header_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        logo_label = ctk.CTkLabel(header_frame, text="DefectScan Pro", font=("Roboto", 20, "bold"), text_color="#3b82f6")
        logo_label.grid(row=0, column=0, sticky="w", padx=(20, 0), pady=15)

        subtitle_label = ctk.CTkLabel(header_frame, text="Automated Quality Control", font=("Roboto", 10), text_color="#666")
        subtitle_label.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=15)

        welcome_label = ctk.CTkLabel(header_frame, text="Welcome, User", font=("Roboto", 14), text_color="#333")
        welcome_label.grid(row=0, column=2, columnspan=2, sticky="e", padx=(0, 10))

        logout_button = ctk.CTkButton(header_frame, text="Logout", width=80, height=30, fg_color="transparent",
                                      text_color="#e74c3c", hover_color="#fbe4e4", font=("Roboto", 12, "bold"),
                                      border_width=1, border_color="#e74c3c",command=self.log_out)
        logout_button.grid(row=0, column=4, sticky="e", padx=(0, 20), pady=10)

    def _create_main_content(self):
        main_content_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        main_content_frame.grid_columnconfigure(0, weight=1)

        # Overview
        dashboard_overview_frame = ctk.CTkFrame(main_content_frame, fg_color="transparent")
        dashboard_overview_frame.pack(fill="x", pady=(0, 20))
        dashboard_overview_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(dashboard_overview_frame, text="Dashboard Overview", font=("Roboto", 22, "bold"), text_color="#333").grid(row=0, column=0, sticky="w", columnspan=3)
        ctk.CTkLabel(dashboard_overview_frame, text="Monitor your defect detection analytics and manage image analysis", font=("Roboto", 12), text_color="#666").grid(row=1, column=0, sticky="w", columnspan=3, pady=(0, 15))

        self._create_overview_card(dashboard_overview_frame, 2, 0, "Images Analyzed", "247", "📄", "blue")
        self._create_overview_card(dashboard_overview_frame, 2, 1, "Defects Detected", "89", "🔲", "orange")
        self._create_overview_card(dashboard_overview_frame, 2, 2, "Success Rate", "94.2%", "📈", "green")

        # Action cards
        action_cards_frame = ctk.CTkFrame(main_content_frame, fg_color="transparent")
        action_cards_frame.pack(fill="x", pady=(0, 20))
        action_cards_frame.grid_columnconfigure((0, 1), weight=1)

        upload_card = ctk.CTkFrame(action_cards_frame, fg_color="white", corner_radius=10)
        upload_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        upload_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(upload_card, text="Upload New Image", font=("Roboto", 16, "bold"), text_color="#333").pack(pady=(20, 5))
        ctk.CTkLabel(upload_card, text="Upload and analyze images for defect detection", font=("Roboto", 12), text_color="#666").pack(pady=(0, 15))
        ctk.CTkButton(upload_card, text="Start Analysis", width=200, height=40, font=("Roboto", 14, "bold"), command=self.open_upload).pack(pady=(0, 20))

        history_card = ctk.CTkFrame(action_cards_frame, fg_color="white", corner_radius=10)
        history_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        history_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(history_card, text="View Reports History", font=("Roboto", 16, "bold"), text_color="#333").pack(pady=(20, 5))
        ctk.CTkLabel(history_card, text="Browse previous analysis reports and results", font=("Roboto", 12), text_color="#666").pack(pady=(0, 15))
        ctk.CTkButton(history_card, text="View History", width=200, height=40, font=("Roboto", 14, "bold"), command=self.open_history).pack(pady=(0, 20))

        # Recent activity
        recent_activity_frame = ctk.CTkFrame(main_content_frame, fg_color="white", corner_radius=10)
        recent_activity_frame.pack(fill="x", pady=(0, 20))
        recent_activity_frame.grid_columnconfigure(0, weight=1)
        recent_activity_frame.grid_columnconfigure(1, weight=0)
        ctk.CTkLabel(recent_activity_frame, text="Recent Activity", font=("Roboto", 22, "bold"), text_color="#333").grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10), columnspan=2)

        self._create_activity_item(recent_activity_frame, 1, "Analyzed circuit_board_001.jpg", "3 hours ago", "3 defects detected", "#e74c3c")
        self._create_activity_item(recent_activity_frame, 2, "Generated report for batch_analysis_14", "5 hours ago", "Report downloaded", "#2ecc71")
        self._create_activity_item(recent_activity_frame, 3, "Analyzed metal_component_052.jpg", "1 day ago", "No defects found", "#2ecc71")

    def _create_overview_card(self, parent_frame, row, column, title, value, icon, icon_color_key):
        card_frame = ctk.CTkFrame(parent_frame, fg_color="white", corner_radius=10)
        card_frame.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
        card_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(card_frame, text=title, font=("Roboto", 14), text_color="#666").grid(row=0, column=0, sticky="nw", padx=20, pady=(20, 0))
        ctk.CTkLabel(card_frame, text=value, font=("Roboto", 28, "bold"), text_color="#333").grid(row=1, column=0, sticky="nw", padx=20, pady=(5, 20))

        icon_color = {"blue": "#3b82f6", "orange": "#f39c12", "green": "#2ecc71"}.get(icon_color_key, "#888")
        icon_label = ctk.CTkLabel(card_frame, text=icon, font=("Segoe UI Emoji", 36), text_color=icon_color)
        icon_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=(0, 20))

    def _create_activity_item(self, parent_frame, row, description, time_ago, status_text, status_color):
        item_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        item_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        item_frame.grid_columnconfigure(0, weight=1)
        item_frame.grid_columnconfigure(1, weight=0)

        desc_label = ctk.CTkLabel(item_frame, text=description, font=("Roboto", 14), text_color="#333")
        desc_label.grid(row=0, column=0, sticky="w")
        time_label = ctk.CTkLabel(item_frame, text=time_ago, font=("Roboto", 10), text_color="#999")
        time_label.grid(row=1, column=0, sticky="w")

        status_badge = ctk.CTkLabel(item_frame, text=f" {status_text} ", corner_radius=5, fg_color=status_color, text_color="white", font=("Roboto", 10, "bold"))
        status_badge.grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))

    def open_history(self):
        try:
            subprocess.Popen([sys.executable, "UI/ReportHistory.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception:
            subprocess.Popen([sys.executable, "UI/ReportHistory.py"])

    def open_upload(self):
        try:
            subprocess.Popen([sys.executable, "UI/Uploadfile.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        except Exception:
            subprocess.Popen([sys.executable, "UI/Uploadfile.py"])


if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
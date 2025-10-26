# 


import customtkinter as ctk
import subprocess
import sys, os

# Ensure project root is on sys.path so absolute imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.auth_service import login_user

# Read optional created message passed from sign-up screen
CREATED_MSG = None
for _arg in sys.argv[1:]:
    if _arg.startswith("--created="):
        CREATED_MSG = _arg.split("=", 1)[1]
        break


class LoginApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure main window
        self.title("Login")
        self.geometry("450x650")
        self.configure(fg_color="#f0f2f5")

        # Main login frame
        self.login_frame = ctk.CTkFrame(
            master=self, width=350, height=500, corner_radius=10, fg_color="white"
        )
        self.login_frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
        self.login_frame.pack_propagate(False)

        # UI setup
        self.create_widgets()

        # If navigated from signup with success message, show it
        # if CREATED_MSG:
        #     try:
        #         self.status_label.configure(text=CREATED_MSG, text_color="#10b981")
        #     except Exception:
        #         pass

    def create_widgets(self):
        # Logo placeholder
        logo_icon = ctk.CTkLabel(
            master=self.login_frame,
            text="",
            fg_color="#3b82f6",
            width=50,
            height=50,
            corner_radius=10,
        )
        logo_icon.pack(pady=(40, 5))

        # Title + subtitle
        title_label = ctk.CTkLabel(
            master=self.login_frame, text="Welcome Back", font=("Roboto", 24, "bold")
        )
        title_label.pack(pady=(10, 0))

        subtitle_label = ctk.CTkLabel(
            master=self.login_frame,
            text="Sign in to your account",
            font=("Roboto", 12),
        )
        subtitle_label.pack(pady=(0, 20))

        # Username
        username_label = ctk.CTkLabel(
            master=self.login_frame, text="Username", font=("Roboto", 12, "bold")
        )
        username_label.pack(anchor=ctk.W, padx=25)

        self.username_entry = ctk.CTkEntry(
            master=self.login_frame,
            placeholder_text="Enter your username",
            width=300,
            height=40,
        )
        self.username_entry.pack(pady=(5, 15))

        # Password
        password_label = ctk.CTkLabel(
            master=self.login_frame, text="Password", font=("Roboto", 12, "bold")
        )
        password_label.pack(anchor=ctk.W, padx=25)

        self.password_entry = ctk.CTkEntry(
            master=self.login_frame,
            placeholder_text="Enter your password",
            show="•",
            width=300,
            height=40,
        )
        self.password_entry.pack(pady=(5, 25))

        # Status label
        self.status_label = ctk.CTkLabel(
            master=self.login_frame,
            text="",
            text_color="#ef4444",
            font=("Roboto", 11),
        )
        self.status_label.pack(pady=(0, 5))

        # Sign in button
        sign_in_button = ctk.CTkButton(
            master=self.login_frame,
            text="Sign In",
            command=self.sign_in_event,
            width=300,
            height=40,
        )
        sign_in_button.pack(pady=(0, 15))

        # Signup link
        signup_link = ctk.CTkButton(
            master=self.login_frame,
            text="Don't have an account? Sign up",
            command=self.go_to_sign_in,
            fg_color="transparent",
            text_color="#3b82f6",
            hover=False,
        )
        signup_link.pack(pady=(0, 20))

    def sign_in_event(self):
        """Validate credentials against MySQL and open dashboard on success."""
        username = self.username_entry.get()
        password = self.password_entry.get()
        ok, msg = login_user(username, password)
        if ok:
            self.status_label.configure(text="", text_color="#10b981")
            try:
                subprocess.Popen(
                    [sys.executable, "UI/Dashboard.py"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )
            except Exception:
                subprocess.Popen([sys.executable, "UI/Dashboard.py"])
            self.withdraw()  # Hide login window
        else:
            self.status_label.configure(text=msg, text_color="#ef4444")

    def go_to_sign_in(self):
        """Go to sign in page (sign_in.py)."""
        subprocess.Popen([sys.executable, "UI/sign_in.py"])
        self.withdraw()  # Hide login window


if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = LoginApp()
    app.mainloop()

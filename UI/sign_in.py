import customtkinter as ctk
import subprocess
import sys, os

# Ensure project root is on sys.path so absolute imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.auth_service import signup_user

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Function to handle sign up / account creation
def sign_up_event():
    username = username_entry.get()
    password = password_entry.get()
    confirm = confirmpassword_label_entry.get()
    if not username or not password:
        status_label.configure(text="Username and password are required.", text_color="#ef4444")
        return
    if password != confirm:
        status_label.configure(text="Passwords do not match.", text_color="#ef4444")
        return
    ok, msg = signup_user(username, password)
    if ok:
        status_label.configure(text=msg, text_color="#10b981")
        # Navigate to login with a success banner after short delay
        app.after(700, lambda: go_to_sign_in(msg))
    else:
        status_label.configure(text=msg, text_color="#ef4444")

def go_to_sign_in(message=None):
    """Run the sign_in.py file when clicked."""
    # Use the current python executable to run the other file
    args = [sys.executable, "UI/login.py"]
    if message:
        # Pass a created message to login screen
        args.append(f"--created={message}")
    subprocess.Popen(args)
    app.withdraw()  # hides the current window

    # app.destroy()  # optional: close current window
# Main window
app = ctk.CTk()
app.title("Create Account")
app.geometry("450x650")
app.configure(fg_color="#f0f2f5")

# Main frame
signup_frame = ctk.CTkFrame(master=app, width=350, height=650, corner_radius=10, fg_color="white")
signup_frame.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)
signup_frame.pack_propagate(False)

# Logo placeholder
logo_icon = ctk.CTkLabel(master=signup_frame, text="", fg_color="#3b82f6", width=50, height=50, corner_radius=10)
logo_icon.pack(pady=(30, 5))

# Title
title_label = ctk.CTkLabel(master=signup_frame, text="Create Account", font=("Roboto", 24, "bold"))
title_label.pack(pady=(10, 0))

# Subtitle
subtitle_label = ctk.CTkLabel(master=signup_frame, text="Fill in your details to sign up", font=("Roboto", 12))
subtitle_label.pack(pady=(0, 20))

# Username
username_label = ctk.CTkLabel(master=signup_frame, text="Username", font=("Roboto", 12, "bold"))
username_label.pack(anchor=ctk.W, padx=25)
username_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Enter your username", width=300, height=40)
username_entry.pack(pady=(5, 15))

# Password
password_label = ctk.CTkLabel(master=signup_frame, text="Password", font=("Roboto", 12, "bold"))
password_label.pack(anchor=ctk.W, padx=25)
password_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Enter your password", show="•", width=300, height=40)
password_entry.pack(pady=(5, 15))

# confirm Password
confirmpassword_label = ctk.CTkLabel(master=signup_frame, text="confirm_password", font=("Roboto", 12, "bold"))
confirmpassword_label.pack(anchor=ctk.W, padx=25)
confirmpassword_label_entry = ctk.CTkEntry(master=signup_frame, placeholder_text="Enter your password again", width=300, height=40)
confirmpassword_label_entry.pack(pady=(5, 15))




# Sign Up button
status_label = ctk.CTkLabel(master=signup_frame, text="", font=("Roboto", 11))
status_label.pack(pady=(0, 5))

sign_up_button = ctk.CTkButton(master=signup_frame, text="Create Account", command=sign_up_event, width=300, height=40)
sign_up_button.pack(pady=(0, 15))



signup_link = ctk.CTkButton(master=signup_frame, text="Already have an account? Sign in",command=go_to_sign_in, fg_color="transparent", text_color="#3b82f6", hover=False)
signup_link.pack(pady=(0, 20))
# Run the app
app.mainloop()

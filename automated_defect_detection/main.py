import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from UI.login import LoginApp
from models import User, Image, Report, Defect
from database_manager import init_db, save_user, load_user, save_image, save_defects, save_report
from image_processor import ImageProcessor
import os
import uuid
import datetime



import customtkinter as ctk
import sys, os

# Ensure project root is on sys.path so absolute imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from automated_defect_detection.auth_service import login_user

# Set the appearance mode and default color theme
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# Database aur directories ko initialize karen
init_db()
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("processed_images", exist_ok=True)
os.makedirs("reports", exist_ok=True)

class DefectDetectionSystem:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.current_user = None

    def login(self, username, password):
        user_data = load_user(username)
        if user_data and user_data['passwordHash'] == password:
            self.current_user = User(user_data['username'], user_data['passwordHash'])
            self.current_user.userID = user_data['userID'] # Set the correct ID from DB
            return True
        return False
        
    def signup(self, username, password):
        # A simple signup function to create a new user in the DB
        existing_user = load_user(username)
        if existing_user:
            return "Error: User already exists."
        
        new_user = User(username, password)
        save_user(new_user)
        return "Signup successful. Please log in."

    def upload_and_process_image(self, image_file_path):
        if not self.current_user:
            return "Error: User not logged in."

        # 1. Save original image file to local storage
        original_filename = os.path.basename(image_file_path)
        original_path = os.path.join("uploaded_images", original_filename)
        # In a real app, you would copy the uploaded file here.
        # For this example, let's assume it's already there.

        # 2. Save Image metadata to the database
        image = Image(self.current_user.userID, original_filename, original_path)
        save_image(image)

        # 3. Process the image and detect defects
        defects_list = self.image_processor.detect(image)

        # 4. Highlight defects and save processed image
        processed_path = self.image_processor.highlight(image, defects_list)
        image.processedPath = processed_path

        # 5. Save defects metadata to the database
        save_defects(defects_list)

        # 6. Generate and save report to the database
        report_id = str(uuid.uuid4())
        report_path = os.path.join("reports", f"{report_id}.pdf")
        report = Report(report_id, image.imageID, len(defects_list), report_path)
        save_report(report)
        
        return f"Processing complete. Defects detected: {len(defects_list)}. Report saved to {report.reportPath}"

# Example Usage    
if __name__ == "__main__":
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = LoginApp()
    app.mainloop()


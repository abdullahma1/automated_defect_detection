import json
import os

DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
IMAGES_FILE = os.path.join(DATA_DIR, "images_metadata.json")
DEFECTS_FILE = os.path.join(DATA_DIR, "defects_metadata.json")
REPORTS_FILE = os.path.join(DATA_DIR, "reports_metadata.json")

def _load_data(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        return json.load(f)

def _save_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_user(user_data):
    users = _load_data(USERS_FILE)
    users.append(user_data.__dict__)
    _save_data(USERS_FILE, users)

def load_user(username):
    users = _load_data(USERS_FILE)
    for user in users:
        if user['username'] == username:
            return user
    return None

# Similarly, write functions for images, defects, and reports
# Example: save_image_metadata, load_image_metadata, save_defects, etc.
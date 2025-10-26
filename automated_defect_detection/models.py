import uuid
import datetime

class User:
    def __init__(self, username, password_hash):
        self.userID = str(uuid.uuid4())
        self.username = username
        self.passwordHash = password_hash

class Image:
    def __init__(self, user_id, filename, original_path, processed_path=None, status="uploaded"):
        self.imageID = str(uuid.uuid4())
        self.userID = user_id
        self.filename = filename
        self.uploadDate = datetime.datetime.now().isoformat()
        self.originalPath = original_path
        self.processedPath = processed_path
        self.status = status

class Defect:
    def __init__(self, image_id, defect_type, bounding_box, confidence):
        self.defectID = str(uuid.uuid4())
        self.imageID = image_id
        self.type = defect_type
        self.boundingBox = bounding_box
        self.confidence = confidence

class Report:
    def __init__(self, report_id, image_id, defect_count, report_path):
        self.reportID = report_id
        self.imageID = image_id
        self.reportDate = datetime.datetime.now().isoformat()
        self.defectCount = defect_count
        self.reportPath = report_path












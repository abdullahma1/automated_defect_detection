import os
import uuid
import datetime
from typing import Dict, Any, List, Optional

from .models import User, Image, Defect, Report
from .database_manager import (
    init_db,
    save_user,
    load_user,
    save_image,
    save_defects,
    save_report,
)


class BackendService:
    def __init__(self) -> None:
        init_db()
        os.makedirs("uploaded_images", exist_ok=True)
        os.makedirs("processed_images", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        self.current_user: Optional[Dict[str, Any]] = None

    def set_current_user(self, username: str) -> bool:
        user_data = load_user(username)
        if user_data:
            self.current_user = user_data
            return True
        return False

    def ensure_user(self, username: str, password_hash: str) -> Dict[str, Any]:
        user = load_user(username)
        if user:
            return user
        # Create a minimal user if it doesn't exist (assumes password_hash is already hashed)
        u = User(username, password_hash)
        save_user(u)
        return {"userID": u.userID, "username": u.username, "passwordHash": u.passwordHash}

    def process_image(self, image_file_path: str, detected_defects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not self.current_user:
            return {"ok": False, "error": "User not logged in."}

        # 1) Save image metadata
        original_filename = os.path.basename(image_file_path)
        original_path = os.path.join("uploaded_images", original_filename)
        # Note: Not copying file here; UI handles file selection. Store metadata only.

        img = Image(self.current_user["userID"], original_filename, original_path)
        # Ensure MySQL DATETIME friendly value
        img.uploadDate = datetime.datetime.now()
        img.processedPath = os.path.join("processed_images", original_filename)
        img.status = "processed"
        save_image(img)

        # 2) Use provided detected defects (from model) or default to none
        defects_dicts = detected_defects or []
        defects: List[Defect] = []
        for d in defects_dicts:
            defects.append(
                Defect(
                    image_id=img.imageID,
                    defect_type=d.get("type", "Unknown"),
                    bounding_box=d.get("boundingBox", [0, 0, 0, 0]),
                    confidence=float(d.get("confidence", 0.0)),
                )
            )

        # 3) Persist defects
        save_defects(defects)

        # 4) Create and persist report
        report_id = str(uuid.uuid4())
        report_path = os.path.join("reports", f"{report_id}.json")
        rpt = Report(report_id, img.imageID, len(defects), report_path)
        # Ensure MySQL DATETIME friendly value
        rpt.reportDate = datetime.datetime.now()
        save_report(rpt)

        # 5) Write a JSON report file (friendly for UI)
        try:
            import json

            payload = {
                "reportID": rpt.reportID,
                "image": {
                    "imageID": img.imageID,
                    "filename": img.filename,
                    "originalPath": img.originalPath,
                    "processedPath": img.processedPath,
                },
                "reportDate": rpt.reportDate if isinstance(rpt.reportDate, str) else rpt.reportDate.isoformat(),
                "defectCount": rpt.defectCount,
                "defects": [
                    {
                        "defectID": d.defectID,
                        "type": d.type,
                        "confidence": d.confidence,
                        "boundingBox": d.boundingBox,
                    }
                    for d in defects
                ],
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            "ok": True,
            "imageID": img.imageID,
            "reportID": rpt.reportID,
            "defectCount": len(defects),
            "reportPath": report_path,
        }

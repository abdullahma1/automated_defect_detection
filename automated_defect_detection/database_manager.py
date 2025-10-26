import mysql.connector
from mysql.connector import Error
import json
try:
    from .connection import get_db_connection, db_config
except ImportError:
    from connection import get_db_connection, db_config

def init_db():
    conn = None
    try:
        # Create database if it does not exist
        temp_config = db_config.copy()
        temp_config.pop('database')
        temp_conn = mysql.connector.connect(**temp_config)
        cursor = temp_conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            
            # Create the 'users' table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    userID VARCHAR(255) PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    passwordHash VARCHAR(255) NOT NULL
                )
            ''')
            
            # Create the 'images' table with a foreign key to 'users'
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    imageID VARCHAR(255) PRIMARY KEY,
                    userID VARCHAR(255),
                    filename VARCHAR(255),
                    uploadDate DATETIME,
                    originalPath VARCHAR(255),
                    processedPath VARCHAR(255),
                    status VARCHAR(50),
                    FOREIGN KEY(userID) REFERENCES users(userID)
                )
            ''')

            # Create the 'defects' table with a foreign key to 'images'
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS defects (
                    defectID VARCHAR(255) PRIMARY KEY,
                    imageID VARCHAR(255),
                    type VARCHAR(100),
                    boundingBox TEXT,  -- JSON string for bounding box list
                    confidence FLOAT,
                    FOREIGN KEY(imageID) REFERENCES images(imageID)
                )
            ''')
            
            # Create the 'reports' table with a foreign key to 'images'
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    reportID VARCHAR(255) PRIMARY KEY,
                    imageID VARCHAR(255),
                    reportDate DATETIME,
                    defectCount INT,
                    reportPath VARCHAR(255),
                    FOREIGN KEY(imageID) REFERENCES images(imageID)
                )
            ''')
            conn.commit()
            print("Saare tables safalta-poorvak bana diye gaye hain.") # All tables have been created successfully.
    except Error as e:
        print(f"Database ko initialize karne mein error: {e}") # Error initializing database: {e}
    finally:
        if conn and conn.is_connected():
            conn.close()

def save_user(user):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (userID, username, passwordHash) VALUES (%s, %s, %s)
            ''', (user.userID, user.username, user.passwordHash))
            conn.commit()
            print("User data save ho gaya.") # User data saved.
        except Error as e:
            print(f"User ko save karne mein error: {e}") # Error saving user.
        finally:
            cursor.close()
            conn.close()

def load_user(username):
    conn = get_db_connection()
    user_data = None
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user_data = cursor.fetchone()
        except Error as e:
            print(f"User ko load karne mein error: {e}") # Error loading user.
        finally:
            cursor.close()
            conn.close()
    return user_data

def save_image(image):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO images (imageID, userID, filename, uploadDate, originalPath, processedPath, status) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (image.imageID, image.userID, image.filename, image.uploadDate, image.originalPath, image.processedPath, image.status))
            conn.commit()
            print("Image metadata save ho gaya.") # Image metadata saved.
        except Error as e:
            print(f"Image ko save karne mein error: {e}") # Error saving image.
        finally:
            cursor.close()
            conn.close()

def save_defects(defects):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            sql = '''INSERT INTO defects (defectID, imageID, type, boundingBox, confidence) VALUES (%s, %s, %s, %s, %s)'''
            defect_data = [(d.defectID, d.imageID, d.type, json.dumps(d.boundingBox), d.confidence) for d in defects]
            cursor.executemany(sql, defect_data)
            conn.commit()
            print("Defects save ho gaye.") # Defects saved.
        except Error as e:
            print(f"Defects ko save karne mein error: {e}") # Error saving defects.
        finally:
            cursor.close()
            conn.close()

def save_report(report):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO reports (reportID, imageID, reportDate, defectCount, reportPath) VALUES (%s, %s, %s, %s, %s)
            ''', (report.reportID, report.imageID, report.reportDate, report.defectCount, report.reportPath))
            conn.commit()
            print("Report metadata save ho gaya.") # Report metadata saved.
        except Error as e:
            print(f"Report ko save karne mein error: {e}") # Error saving report.
        finally:
            cursor.close()
            conn.close()

def fetch_recent_reports(limit=10):
    conn = get_db_connection()
    rows = []
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute('''
                SELECT r.reportID, r.imageID, r.reportDate, r.defectCount, r.reportPath,
                       i.filename, i.userID
                FROM reports r
                LEFT JOIN images i ON i.imageID = r.imageID
                ORDER BY r.reportDate DESC
                LIMIT %s
            ''', (int(limit),))
            rows = cursor.fetchall() or []
        except Error as e:
            print(f"Recent reports fetch karne mein error: {e}")
        finally:
            cursor.close()
            conn.close()
    return rows

def fetch_report_details(report_id):
    conn = get_db_connection()
    details = None
    if conn:
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute('''
                SELECT r.reportID, r.imageID, r.reportDate, r.defectCount, r.reportPath,
                       i.filename, i.uploadDate, i.originalPath, i.processedPath, i.userID
                FROM reports r
                LEFT JOIN images i ON i.imageID = r.imageID
                WHERE r.reportID = %s
            ''', (report_id,))
            details = cursor.fetchone()
            if details:
                # Fetch defects tied to the image
                cursor.execute('''
                    SELECT defectID, imageID, type, boundingBox, confidence
                    FROM defects
                    WHERE imageID = %s
                ''', (details['imageID'],))
                defects = cursor.fetchall() or []
                # Ensure boundingBox is parsed JSON
                for d in defects:
                    try:
                        d['boundingBox'] = json.loads(d.get('boundingBox') or '[]')
                    except Exception:
                        pass
                details['defects'] = defects
        except Error as e:
            print(f"Report details fetch karne mein error: {e}")
        finally:
            cursor.close()
            conn.close()
    return details

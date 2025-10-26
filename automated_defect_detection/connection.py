import mysql.connector
from mysql.connector import Error

# Replace these placeholders with your actual MySQL credentials
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'defect_detection'
}

def get_db_connection():
    """
    Establishes and returns a connection to the MySQL database.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            print("Successfully connected to MySQL database.")
            return conn
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

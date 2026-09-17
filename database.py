import sqlite3


DATABASE = "attendance.db"


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            student_type TEXT,
            face_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Backward compatibility: if students table already existed
    # (from before this column was added), add it in place instead
    # of losing existing registrations.
    existing_columns = [
        row["name"]
        for row in cursor.execute("PRAGMA table_info(students)")
    ]

    if "student_type" not in existing_columns:
        cursor.execute(
            "ALTER TABLE students ADD COLUMN student_type TEXT"
        )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


if __name__ == "__main__":
    create_tables()
    print("Database created successfully!")
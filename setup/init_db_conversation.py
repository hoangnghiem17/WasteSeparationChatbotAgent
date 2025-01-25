import sqlite3

def initialize_database(db_path):
    """
    Initializes the SQLite database with the conversation_logs table.
    This table tracks user and agent messages, storing text and image data separately.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the conversation_logs table with separate columns for text and image data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversation_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'agent')),
        message_text TEXT, -- Column for storing text messages
        message_image TEXT, -- Column for storing image file paths
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    initialize_database("database/conversation.db")

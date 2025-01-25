# agent/logger.py
import sqlite3
import logging
from datetime import datetime

DB_PATH = "database/conversation.db"

def log_message(conversation_id: str, role: str, message_text: str = None, message_image: str = None):
    """
    Logs a message to the SQLite database.

    Args:
        conversation_id (str): Unique ID for the conversation.
        role (str): The sender of the message ('user' or 'agent').
        message_text (str, optional): The text content of the message.
        message_image (str, optional): The file path of the uploaded image.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO conversation_logs (conversation_id, role, message_text, message_image, timestamp)
        VALUES (?, ?, ?, ?, ?)
        ''', (conversation_id, role, message_text, message_image, datetime.utcnow()))
        
        conn.commit()
        conn.close()
        logging.info(f"Logged message to conversation {conversation_id}.")
    except Exception as e:
        logging.error(f"Failed to log message: {e}")

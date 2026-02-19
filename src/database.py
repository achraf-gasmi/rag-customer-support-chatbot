import sqlite3
import os

DB_PATH = "logs/interactions.db"

def init_db():
    """Create the interactions table if it doesn't exist."""
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            query_text       TEXT NOT NULL,
            response_text    TEXT NOT NULL,
            source           TEXT NOT NULL,
            category         TEXT,
            intent           TEXT,
            confidence_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(timestamp, query_text, response_text, source, category, intent, confidence_score):
    """Log a single interaction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions 
        (timestamp, query_text, response_text, source, category, intent, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, query_text, response_text, source, category, intent, confidence_score))
    conn.commit()
    conn.close()

def get_all_interactions():
    """Retrieve all logged interactions."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
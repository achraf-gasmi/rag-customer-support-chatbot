import sqlite3
import os

DB_PATH = "logs/interactions.db"

def init_db():
    os.makedirs("logs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Interactions table
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

    # Dynamic graph edges table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source_cat  TEXT NOT NULL,
            target_cat  TEXT NOT NULL,
            weight      INTEGER DEFAULT 1,
            UNIQUE(source_cat, target_cat)
        )
    """)

    conn.commit()
    conn.close()

def log_interaction(timestamp, query_text, response_text, source, category, intent, confidence_score):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interactions
        (timestamp, query_text, response_text, source, category, intent, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, query_text, response_text, source, category, intent, confidence_score))
    conn.commit()

    # Update dynamic graph — find last category and create/strengthen edge
    cursor.execute("""
        SELECT category FROM interactions
        ORDER BY id DESC LIMIT 2
    """)
    rows = cursor.fetchall()
    if len(rows) == 2:
        prev_cat = rows[1][0]
        curr_cat = rows[0][0]
        if prev_cat and curr_cat and prev_cat != curr_cat and prev_cat != "UNKNOWN" and curr_cat != "UNKNOWN":
            cursor.execute("""
                INSERT INTO graph_edges (source_cat, target_cat, weight)
                VALUES (?, ?, 1)
                ON CONFLICT(source_cat, target_cat)
                DO UPDATE SET weight = weight + 1
            """, (prev_cat, curr_cat))
            conn.commit()
            print(f"[DynamicGraph] Edge strengthened: {prev_cat} → {curr_cat}")

    conn.close()

def get_all_interactions():
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interactions ORDER BY timestamp DESC")
    rows   = cursor.fetchall()
    conn.close()
    return rows

def get_dynamic_edges():
    """Get learned edges from interaction history."""
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT source_cat, target_cat, weight FROM graph_edges ORDER BY weight DESC")
    rows   = cursor.fetchall()
    conn.close()
    return rows
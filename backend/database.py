import sqlite3
import json
import datetime
import uuid

DB_NAME = "dds_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # History table
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id TEXT PRIMARY KEY, 
                  timestamp TEXT, 
                  type TEXT, 
                  filename TEXT, 
                  result TEXT)''')
    
    # Feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (scan_id TEXT, 
    return scan_id

def get_history():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "type": row["type"],
            "filename": row["filename"],
            "result": json.loads(row["result"])
        })
    conn.close()
    return history

def save_feedback(scan_id, rating, comments):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO feedback (scan_id, rating, comments) VALUES (?, ?, ?)",
              (scan_id, rating, comments))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT result FROM history")
    rows = c.fetchall()
    conn.close()

    total_scans = len(rows)
    threats_detected = 0
    total_confidence = 0

    for row in rows:
        try:
            result = json.loads(row["result"])
            if result.get("is_fake"):
                threats_detected += 1
            total_confidence += result.get("confidence_score", 0)
        except:
            pass
            
    avg_confidence = (total_confidence / total_scans) if total_scans > 0 else 0
    authentic_media = total_scans - threats_detected

    return {
        "total_scans": total_scans,
        "threats_detected": threats_detected,
        "authentic_media": authentic_media,
        "avg_confidence": round(avg_confidence, 1)
    }

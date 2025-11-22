import sqlite3
import json
import datetime
import uuid
import numpy as np # Import numpy globally as it's used in save_scan_result

DB_NAME = "dds_history.db"

def init_db():
    """Initializes the SQLite database, creating history and feedback tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # History table to store scan results
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id TEXT PRIMARY KEY, 
                  timestamp TEXT, 
                  type TEXT, 
                  filename TEXT, 
                  result TEXT)''')
    
    # Feedback table to store user feedback on scans
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (scan_id TEXT, 
                  rating INTEGER, 
                  comments TEXT,
                  FOREIGN KEY(scan_id) REFERENCES history(id))''')
    
    conn.commit()
    conn.close()

def _convert_to_python_types(obj):
    """
    Recursively converts NumPy types within an object (dict, list, or scalar)
    to standard Python types for JSON serialization.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_python_types(item) for item in obj]
    return obj

def save_scan_result(media_type, filename, result_dict):
    """
    Saves a scan result to the history table.
    Automatically converts NumPy types in result_dict to Python types for JSON serialization.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Convert any numpy types in the result dictionary to standard Python types
    # to ensure proper JSON serialization.
    result_dict_clean = _convert_to_python_types(result_dict)
    
    scan_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    result_json = json.dumps(result_dict_clean)
    
    c.execute("INSERT INTO history (id, timestamp, type, filename, result) VALUES (?, ?, ?, ?, ?)",
              (scan_id, timestamp, media_type, filename, result_json))
    conn.commit()
    conn.close()
    return scan_id

def get_history():
    """
    Retrieves the 20 most recent scan results from the history table.
    The 'result' field, stored as JSON, is parsed back into a dictionary.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    c = conn.cursor()
    
    c.execute("SELECT id, timestamp, type, filename, result FROM history ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    
    history = []
    for row in rows:
        try:
            result_data = json.loads(row["result"])
        except json.JSONDecodeError:
            result_data = {"error": "Failed to parse result JSON"}
            
        history.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "type": row["type"],
            "filename": row["filename"],
            "result": result_data
        })
    conn.close()
    return history

def save_feedback(scan_id, rating, comments):
    """
    Saves user feedback for a specific scan ID to the feedback table.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("INSERT INTO feedback (scan_id, rating, comments) VALUES (?, ?, ?)",
              (scan_id, rating, comments))
    conn.commit()
    conn.close()

def get_stats():
    """
    Calculates and returns various statistics based on the scan history,
    including total scans, threats detected, authentic media count, and average confidence.
    """
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
        except (json.JSONDecodeError, AttributeError):
            # Handle cases where result might not be valid JSON or expected dict structure
            pass
            
    avg_confidence = (total_confidence / total_scans) if total_scans > 0 else 0
    authentic_media = total_scans - threats_detected

    return {
        "total_scans": total_scans,
        "threats_detected": threats_detected,
        "authentic_media": authentic_media,
        "avg_confidence": round(avg_confidence, 1)
    }

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import shutil
import os
import datetime
import sqlite3
from fastapi.responses import StreamingResponse
from database import init_db, save_scan_result, get_history, save_feedback, get_stats
from models import detect_image, detect_text, detect_audio, detect_video

app = FastAPI(title="Deep Fake Detection System API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
init_db()

class FeedbackRequest(BaseModel):
    scan_id: str
    rating: int
    comments: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Deep Fake Detection System API is running"}

@app.get("/stats")
def read_stats():
    return get_stats()

@app.post("/detect/image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save temp file
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run detection
        result = detect_image(temp_file)
        
        # Save to history
        scan_id = save_scan_result("image", file.filename, result)
        result["scan_id"] = scan_id
        
        # Cleanup
        os.remove(temp_file)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/text")
async def analyze_text(text: str = Form(...)):
    try:
        result = detect_text(text)
        scan_id = save_scan_result("text", text[:30] + "...", result)
        result["scan_id"] = scan_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/audio")
async def analyze_audio(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = detect_audio(temp_file)
        scan_id = save_scan_result("audio", file.filename, result)
        result["scan_id"] = scan_id
        
        os.remove(temp_file)
        return result
    except Exception as e:
        print(f"ERROR in audio detection: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = detect_video(temp_file)
        scan_id = save_scan_result("video", file.filename, result)
        result["scan_id"] = scan_id
        
        os.remove(temp_file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_scan_history():
    return get_history()

from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
import io

@app.get("/certificate/{scan_id}")
def generate_certificate(scan_id: str):
    # Fetch scan details
    conn = sqlite3.connect("dds_history.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE id=?", (scan_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Scan not found")
        
    # Create certificate image
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([20, 20, 780, 580], outline="black", width=5)
    
    # Add text (using default font for simplicity, in real app use custom font)
    # Note: ImageFont.load_default() is small, so we might want to try loading a system font or just keep it simple
    try:
        font_title = ImageFont.truetype("arial.ttf", 40)
        font_text = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        
    draw.text((400, 100), "CERTIFICATE OF AUTHENTICITY", fill="black", anchor="mm", font=font_title)
    draw.text((400, 200), f"This certifies that the media file:", fill="black", anchor="mm", font=font_text)
    draw.text((400, 250), f"{row['filename']}", fill="blue", anchor="mm", font=font_text)
    draw.text((400, 300), f"Has been analyzed by DeepTruth AI and found to be:", fill="black", anchor="mm", font=font_text)
    draw.text((400, 350), "AUTHENTIC", fill="green", anchor="mm", font=font_title)
    draw.text((400, 450), f"Scan ID: {scan_id}", fill="gray", anchor="mm", font=font_text)
    draw.text((400, 500), f"Date: {row['timestamp']}", fill="gray", anchor="mm", font=font_text)
    
    # Save to buffer
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/png")

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    save_feedback(feedback.scan_id, feedback.rating, feedback.comments)
    return {"status": "success", "message": "Feedback received"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

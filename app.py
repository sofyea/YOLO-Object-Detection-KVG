from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="API Deployment untuk Projek DKB 3263 menggunakan FastAPI & YOLOv8",
    version="1.0.0"
)

# Load model YOLOv8
model = YOLO("yolov8n.pt")

class HealthCheck(BaseModel):
    status: str
    model_loaded: str

@app.get("/", response_model=HealthCheck)
def root():
    return HealthCheck(status="API Operational", model_loaded="YOLOv8n Nano")

@app.post("/predict")
async def predict_object(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLOv8 inference
    results = model(img)
    
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4)
            })
            
    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections
    }
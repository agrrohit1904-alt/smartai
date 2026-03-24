import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

print("Loading the Helmet AI model...")
# 🔴 Remember to put your custom model 'best.pt' in the same folder if exploring Render
model = YOLO('yolo26n.pt') 

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the image sent from the Raspberry Pi / Laptop
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run the AI on the image
    results = model(frame, verbose=False)
    
    helmet_detected = False
    boxes_data = []

    # 3. Analyze the results
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Get coordinates to draw the box later on the client
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        boxes_data.append({
            "class_id": class_id,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })

        # Assuming Class 0 is 'Helmet' and confidence > 65%
        if class_id == 0 and confidence < 0.75:
            helmet_detected = True
            break

    # 4. Return the results as JSON
    return JSONResponse(content={
        "helmet_detected": helmet_detected,
        "predictions": boxes_data
    })

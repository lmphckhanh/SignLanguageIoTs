from flask import Flask, request, jsonify
import os, cv2, time, base64
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# --- Cấu hình MongoDB ---
# Kết nối tới MongoDB (chạy tại localhost mặc định)
client = MongoClient("mongodb://localhost:27017/")
db = client["iot_asl_database"]
collection = db["predictions"]

# Load model (Giữ nguyên cấu hình cũ của bạn)
model = tf.keras.models.load_model("asl_model")
CONFIDENCE_THRESHOLD = 0.9
LABELS = {i: chr(65 + i) for i in range(26)}

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame.astype('float32') / 255.0, axis=0)

def upload_to_mongodb(label, frame, confidence):
    try:
        # Chuyển ảnh sang Base64 để lưu vào Mongo
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tạo document để lưu trữ
        document = {
            "text": label,
            "confidence": float(confidence),
            "image": f"data:image/jpeg;base64,{img_b64}",
            "timestamp": datetime.utcnow() # Sử dụng kiểu ISODate của Mongo để dễ query
        }
        
        # Insert vào collection
        result = collection.insert_one(document)
        print(f"📤 MongoDB Saved: {label} - ID: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"❌ MongoDB error: {e}")
        return False

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return "No image part", 400
            
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Dự đoán
        input_tensor = preprocess(frame)
        preds = model.predict(input_tensor, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]
        label = LABELS.get(pred_class, "-")

        if confidence > CONFIDENCE_THRESHOLD:
            # Gọi hàm lưu vào MongoDB thay vì Firebase
            success = upload_to_mongodb(label, frame, confidence)
            if success:
                return "OK", 200
            else:
                return "Database Error", 500
        else:
            return "Low confidence", 204

    except Exception as e:
        print(f"❌ Server error: {e}")
        return "Error", 500

if __name__ == '__main__':
    # host='0.0.0.0' để Raspberry Pi có thể gọi tới IP máy tính này
    app.run(host='0.0.0.0', port=5000)
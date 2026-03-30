import socketio
import uvicorn
import numpy as np
import tensorflow as tf
from fastapi import FastAPI

# 1. Khởi tạo FastAPI và Socket.IO
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# 2. Nạp Model AI (Giả sử bạn đã có file model.h5)
# model = tf.keras.models.load_model('models/sign_language_model.h5')

@sio.event
async def connect(sid, environ):
    print(f">>> [Server] Thiết bị kết nối: {sid}")

@sio.on('send_coords')
async def handle_coords(sid, data):
    """
    Hàm xử lý tọa độ từ Pi gửi lên
    data['data'] là list tọa độ [x1, y1, z1, x2, y2, z2, ...]
    """
    coords = np.array(data['data']).astype('float32')
    print(coords)
    
    # --- BƯỚC PREPROCESSING (Tiền xử lý) ---
    # Giả sử model cần đầu vào là (1, 21, 3) 
    # Bạn cần reshape lại dữ liệu tùy theo cấu trúc Model của bạn
    # try:
    #     # Ví dụ đơn giản: Reshape về đúng định dạng input của Model
    #     input_data = coords.reshape(1, -1) # Tùy chỉnh theo shape của bạn
        
    #     # --- BƯỚC INFERENCE (Dự đoán) ---
    #     # prediction = model.predict(input_data)
    #     # label = np.argmax(prediction)
        
    #     # Giả lập kết quả để test
    #     detected_text = "HELLO" 
        
    #     # 3. Gửi kết quả ngược lại cho Pi hoặc hiển thị lên Dashboard
    #     print(f"Nhận dữ liệu từ {sid} -> Dự đoán: {detected_text}")
    #     await sio.emit('receive_result', {'text': detected_text}, room=sid)
        
    # except Exception as e:
    #     print(f"Lỗi xử lý dữ liệu: {e}")

@sio.event
async def disconnect(sid):
    print(f">>> [Server] Thiết bị ngắt kết nối: {sid}")

if __name__ == "__main__":
    # Chạy Server trên port 8000
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
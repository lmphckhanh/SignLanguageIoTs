import cv2
import mediapipe as mp
import numpy as np
import os
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socketio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Cấu hình đường dẫn ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'hand_landmarker.task')
SERVER = 'http://192.168.1.54:8000'

# --- Khởi tạo Socket.IO ---
sio = socketio.Client()

@sio.event
def connect():
    print(">>> [Pi] Kết nối thành công tới AI Server!")

@sio.on('receive_result')
def on_result(data):
    # Nhận kết quả dự đoán từ Server để hiển thị (nếu cần)
    print(f"Dự đoán từ Server: {data['text']}")

# --- Khởi tạo Detector ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1, # Chỉnh về 1 tay để tăng tốc độ xử lý nếu không cần 2 tay
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

def draw_landmarks_on_image(image, detection_result):
    if not detection_result.hand_landmarks:
        return image
    height, width, _ = image.shape
    for hand_landmarks in detection_result.hand_landmarks:
        for connection in HAND_CONNECTIONS:
            start_lm = hand_landmarks[connection[0]]
            end_lm = hand_landmarks[connection[1]]
            start_point = (int(start_lm.x * width), int(start_lm.y * height))
            end_point = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return image

# --- Kết nối tới Server ---
try:
    sio.connect(SERVER)
except:
    print(">>> [Pi] Không thể kết nối server. Đang chạy offline...")

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp += 1
    
    # Nhận diện Landmarks
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # Nếu có tay, trích xuất và gửi NGAY LẬP TỨC
    if detection_result.hand_landmarks:
        for hand_lms in detection_result.hand_landmarks:
            # Tạo list tọa độ phẳng [x1, y1, z1, x2, y2, z2...]
            coords = []
            for lm in hand_lms:
                coords.extend([lm.x, lm.y, lm.z])
            
            # Gửi dữ liệu frame hiện tại lên server
            if sio.connected:
                sio.emit('send_coords', {'data': coords})

    # Vẽ và hiển thị
    frame = draw_landmarks_on_image(frame, detection_result)
    cv2.imshow("Pi Client - Real-time", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
if sio.connected:
    sio.disconnect()
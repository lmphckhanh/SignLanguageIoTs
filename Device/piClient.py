import cv2
import mediapipe as mp
import numpy as np
import os
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socketio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'hand_landmarker.task')

DATA_FILE = os.path.join(SCRIPT_DIR, 'data_hand.json')

SERVER = 'http://192.168.1.54:8000'

# Initiate Detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

#Hands Visualization on camera
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),    # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón nhẫn
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17) # Ngón út và lòng bàn tay
]

def draw_landmarks_on_image(image, detection_result):

    if not detection_result.hand_landmarks:
        return image

    height, width, _ = image.shape

    for hand_landmarks in detection_result.hand_landmarks:
        # Vẽ các đường nối xương
        for connection in HAND_CONNECTIONS:
            start_lm = hand_landmarks[connection[0]]
            end_lm = hand_landmarks[connection[1]]
            
            start_point = (int(start_lm.x * width), int(start_lm.y * height))
            end_point = (int(end_lm.x * width), int(end_lm.y * height))
            
            cv2.line(image, start_point, end_point, (255, 255, 255), 2) # Đường màu trắng

        # Vẽ các khớp ngón tay
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1) # Chấm màu đỏ

    return image


#---------------MAIN------------------------

sio = socketio.Client()
@sio.event
def connect():
    print(">>> [Pi] Kết nối thành công tới AI Server!")

cap = cv2.VideoCapture(0)
timestamp = 0
collected_data = []

try:
    sio.connect(SERVER)
except:
    print(">>> [Pi] Offline Mode: Đang chạy chế độ kiểm tra cục bộ.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # MediaPipe Tasks yêu cầu chuyển sang RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp += 1
    
    # Nhận diện
    detection_result = detector.detect_for_video(mp_image, timestamp)

    # Vẽ lên màn hình
    frame = draw_landmarks_on_image(frame, detection_result)

    # Trích xuất dữ liệu để lưu JSON
    if detection_result.hand_landmarks:
        for hand_lms in detection_result.hand_landmarks:
            coords = [lm.x for lm in hand_lms] + [lm.y for lm in hand_lms] + [lm.z for lm in hand_lms]
            collected_data.append({"frame": timestamp, "data": coords})

    cv2.imshow("Hand Tracking (Tasks V2)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lưu dữ liệu
with open(DATA_FILE, "w") as f:
    json.dump(collected_data, f)

if sio.connected:
            sio.emit('send_coords', {'data': collected_data})

cap.release()
cv2.destroyAllWindows()
detector.close()
sio.disconnect()
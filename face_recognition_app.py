# File: face_recognition_app.py (เวอร์ชัน Blink Detection + Arduino Control)
import cv2
import dlib
import numpy as np
import pickle
from keras.models import load_model
from scipy.spatial import distance as dist
import time
import serial

# --- 1. การตั้งค่า ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
BLINKS_NEEDED = 2
CONFIDENCE_THRESHOLD = 0.70
RECOGNITION_IMG_SIZE = 64
LIVENESS_IMG_SIZE = 32

# --- การตั้งค่าการเชื่อมต่อกับ ARDUINO ---
ARDUINO_PORT = 'COM3' # <--- แก้ไข Port ตรงนี้
try:
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=9600, timeout=.1)
    print(f"✅ เชื่อมต่อกับ Arduino UNO ที่ {ARDUINO_PORT} สำเร็จ!")
except serial.SerialException:
    arduino = None
    print(f"⚠️ ไม่สามารถเชื่อมต่อกับ Arduino UNO ที่ {ARDUINO_PORT} ได้")

def send_command_to_arduino(command):
    if arduino:
        try:
            arduino.write(bytes(command + "\n", 'utf-8'))
            time.sleep(0.05)
        except serial.SerialException:
            pass

# --- 2. โหลดโมเดลทั้งหมด ---
print("="*50)
print("      Advanced Face Recognition + Per-Person Liveness      ")
print("="*50)
print("\n[ขั้นตอนที่ 1/3] กำลังโหลดโมเดล...")
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    recognition_model = load_model("face_recognition_model.h5")
    with open("label_encoder.pickle", 'rb') as f:
        le = pickle.load(f)
    # --- โหลดโมเดล Liveness ด้วย ---
    liveness_model = load_model("liveness_model.h5")
    with open("liveness_label_encoder.pickle", 'rb') as f:
        liveness_le = pickle.load(f)
    print("  -> ✅ โหลดโมเดลทั้งหมดสำเร็จ!")
except Exception as e:
    input(f"❌ ไม่สามารถโหลดไฟล์โมเดลได้\n   -> Error: {e}\n\nกด Enter เพื่อกลับไปที่เมนู")
    exit()

# --- 3. ฟังก์ชันและตัวแปร ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
face_statuses = {}
next_face_id = 0
AUTHORIZED_USERS = [name.upper() for name in le.classes_]

# --- 4. เริ่มต้นกล้อง ---
print("\n[ขั้นตอนที่ 2/3] กำลังเริ่มต้นกล้องเว็บแคม...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
send_command_to_arduino("UNAUTHORIZED") # เริ่มต้นด้วยไฟแดง

print("\n[ขั้นตอนที่ 3/3] เริ่มการทำงาน...")
print("   -> ทุกคนที่อยู่ในกล้องต้อง 'กะพริบตา 2 ครั้ง' เพื่อยืนยันตัวตน")
while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    rects = detector(rgb_frame, 0)
    current_face_centers = [(r.center().x, r.center().y) for r in rects]
    
    # จัดการสถานะของคนที่หายไป
    disappeared_ids = []
    for face_id, status in face_statuses.items():
        if not any(dist.euclidean(center, status["pos"]) < 75 for center in current_face_centers):
            disappeared_ids.append(face_id)
    for face_id in disappeared_ids:
        del face_statuses[face_id]

    is_authorized_person_in_frame = False

    # วนลูปในแต่ละใบหน้าที่เจอ
    for rect in rects:
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        center = (rect.center().x, rect.center().y)
        
        # จับคู่ใบหน้ากับสถานะที่มีอยู่ หรือสร้างใหม่
        existing_id = next((face_id for face_id, status in face_statuses.items() if dist.euclidean(center, status["pos"]) < 75), None)
        
        if existing_id is not None:
            face_statuses[existing_id]["pos"] = center
            status = face_statuses[existing_id]
        else:
            status = {
                "pos": center, "closed_counter": 0, "blinks": 0,
                "liveness_verified": False, "recognition_done": False,
                "name": "Blink 2 times", "color": (0, 255, 255)
            }
            face_statuses[next_face_id] = status
            next_face_id += 1

        # ถ้ายังไม่ยืนยัน Liveness ด้วยการกะพริบตา
        if not status["liveness_verified"]:
            shape = predictor(rgb_frame, rect)
            shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            ear = (eye_aspect_ratio(shape[lStart:lEnd]) + eye_aspect_ratio(shape[rStart:rEnd])) / 2.0

            if ear < EYE_AR_THRESH:
                status["closed_counter"] += 1
            else:
                if status["closed_counter"] >= EYE_AR_CONSEC_FRAMES:
                    status["blinks"] += 1
                status["closed_counter"] = 0
            
            status["name"] = f"Blinks: {status['blinks']}/{BLINKS_NEEDED}"
            if status["blinks"] >= BLINKS_NEEDED:
                status["liveness_verified"] = True
                print(f"  -> ✅ Blink Liveness Verified for a face.")
        
        # ถ้า Liveness ผ่านแล้ว และยังไม่ได้จดจำใบหน้า
        if status["liveness_verified"] and not status["recognition_done"]:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0: continue

            # --- ด่านที่ 2: ตรวจสอบ Spoof ด้วยโมเดล ---
            liveness_face = cv2.resize(face_roi, (LIVENESS_IMG_SIZE, LIVENESS_IMG_SIZE)).astype("float") / 255.0
            liveness_face = np.expand_dims(liveness_face, axis=0)
            liveness_preds = liveness_model.predict(liveness_face, verbose=0)[0]
            liveness_label = liveness_le.classes_[np.argmax(liveness_preds)]
            
            if liveness_label == "real":
                # --- ถ้าผ่านด่านที่ 2 ถึงจะทำการจดจำใบหน้า ---
                recog_face = cv2.resize(face_roi, (RECOGNITION_IMG_SIZE, RECOGNITION_IMG_SIZE)).astype("float") / 255.0
                recog_face = np.expand_dims(recog_face, axis=0)
                preds = recognition_model.predict(recog_face, verbose=0)[0]
                
                if np.max(preds) > CONFIDENCE_THRESHOLD:
                    name = le.classes_[np.argmax(preds)].upper()
                    status["name"] = name
                    status["color"] = (255, 0, 0) # สีน้ำเงิน
                else:
                    status["name"] = "UNKNOWN"
                    status["color"] = (0, 0, 255) # สีแดง
            else:
                status["name"] = "SPOOF DETECTED"
                status["color"] = (0, 0, 255) # สีแดง
            
            status["recognition_done"] = True
        
        # ตรวจสอบว่าใบหน้าที่ยืนยันตัวตนแล้ว เป็นคนที่ได้รับอนุญาตหรือไม่
        if status.get("recognition_done", False) and status["name"] in AUTHORIZED_USERS:
            is_authorized_person_in_frame = True

        # วาดผลลัพธ์ของใบหน้านี้
        cv2.rectangle(frame, (x, y), (x + w, y + h), status["color"], 2)
        cv2.putText(frame, status["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status["color"], 2)

    # --- สรุปผลและส่งคำสั่งสุดท้ายไปให้ Arduino ---
    if is_authorized_person_in_frame:
        send_command_to_arduino("AUTHORIZED")
    else:
        send_command_to_arduino("UNAUTHORIZED")

    # --- ปรับขนาดและแสดงผล ---
    new_width = 1280
    new_height = 720
    display_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow("Ultimate Secure Face Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n👋 กำลังปิดโปรแกรม...")
send_command_to_arduino("UNAUTHORIZED")
if arduino: arduino.close()
cap.release()
cv2.destroyAllWindows()
import cv2
import dlib
import os
import time
import numpy as np

# --- 1. การตั้งค่า ---
person_name = input("กรุณาป้อนชื่อ (ภาษาอังกฤษเท่านั้น) สำหรับ Dataset นี้: ")
DATASET_PATH = "dataset"
PERSON_PATH = os.path.join(DATASET_PATH, person_name)
os.makedirs(PERSON_PATH, exist_ok=True)

TOTAL_IMAGES_TO_CAPTURE = 50

# --- [สำคัญ] ปรับค่าเหล่านี้หลังจากการดีบัก ---
BLUR_THRESHOLD = 90.0   # ลองเริ่มจากค่าที่ต่ำลงมาหน่อย
POSE_THRESHOLD = 30.0   # ลองเพิ่มค่านี้เพื่อให้ยืดหยุ่นขึ้น
MIN_FACE_WIDTH = 90

# --- 2. โหลดโมเดลของ dlib ---
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except Exception as e:
    print(f"❌ ไม่สามารถโหลดไฟล์ dlib model ได้: {e}")
    exit()

# --- 3. ฟังก์ชัน (ไม่มีการเปลี่ยนแปลง) ---
def is_blurry(image_gray):
    variance = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD, variance

# --- [วางทับฟังก์ชัน get_head_pose เดิมทั้งหมด] ---

def get_head_pose(shape, frame_shape):
    """ประเมินมุมของศีรษะ (พร้อมแก้ไขแกนกลับด้าน)"""
    model_points = np.array([
        (0.0, 0.0, 0.0),             # ปลายจมูก
        (0.0, -330.0, -65.0),        # คาง
        (-225.0, 170.0, -135.0),     # มุมตาซ้ายด้านซ้ายสุด
        (225.0, 170.0, -135.0),      # มุมตาขวาด้านขวาสุด
        (-150.0, -150.0, -125.0),    # มุมปากซ้าย
        (150.0, -150.0, -125.0)      # มุมปากขวา
    ])
    
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),
        (shape.part(8).x,  shape.part(8).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y)
    ], dtype="double")

    focal_length = frame_shape[1]
    center = (frame_shape[1]/2, frame_shape[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
    mat = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(mat)
    
    pitch, yaw, roll = [np.math.radians(_) for _ in euler_angles]
    pitch = np.math.degrees(pitch)
    yaw = np.math.degrees(yaw)

    # --- [ส่วนแก้ไขแกนกลับด้านที่เพิ่มเข้ามา] ---
    # ถ้าค่า pitch อยู่ในโซนที่กลับด้าน ให้พลิกกลับมา
    if pitch > 100:
        pitch = 180 - pitch
    elif pitch < -100:
        pitch = -180 - pitch
    # -------------------------------------------
    
    return abs(yaw) > POSE_THRESHOLD or abs(pitch) > POSE_THRESHOLD, (pitch, yaw)
# --- 4. เริ่มกระบวนการเก็บข้อมูล ---
cap = cv2.VideoCapture(0)
time.sleep(2.0)
image_counter = 0
print("\n[INFO] เริ่มต้นโหมดดีบัก... กรุณามองที่กล้องและสังเกตค่าบนหน้าจอ")

while image_counter < TOTAL_IMAGES_TO_CAPTURE:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    if len(rects) > 0:
        rect = rects[0]
        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        
        if w <= 0 or h <= 0: continue
        face_roi_gray = gray[y:y+h, x:x+w]
        if face_roi_gray.size == 0: continue

        shape = predictor(gray, rect)
        
        blurry, blur_val = is_blurry(face_roi_gray)
        bad_pose, pose_vals = get_head_pose(shape, frame.shape)
        too_small = w < MIN_FACE_WIDTH

        # --- [ส่วนดีบักที่วาดบนจอภาพ] ---
        pose_pitch, pose_yaw = pose_vals
        
        # วาดจุด 6 จุดที่ใช้คำนวณ
        for i in [30, 8, 36, 45, 48, 54]:
            p = shape.part(i)
            cv2.circle(frame, (p.x, p.y), 3, (0, 255, 0), -1)

        # แสดงค่าที่วัดได้ทั้งหมด
        pitch_color = (0, 255, 0) if abs(pose_pitch) <= POSE_THRESHOLD else (0, 0, 255)
        yaw_color = (0, 255, 0) if abs(pose_yaw) <= POSE_THRESHOLD else (0, 0, 255)
        blur_color = (0, 255, 0) if not blurry else (0, 0, 255)
        
        cv2.putText(frame, f"PITCH: {pose_pitch:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_color, 2)
        cv2.putText(frame, f"YAW:   {pose_yaw:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, yaw_color, 2)
        cv2.putText(frame, f"BLUR:  {blur_val:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, blur_color, 2)
        cv2.putText(frame, f"Thresholds: Pose < {POSE_THRESHOLD}, Blur > {BLUR_THRESHOLD}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # -----------------------------------

        if not blurry and not bad_pose and not too_small:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_path = os.path.join(PERSON_PATH, f"{str(image_counter).zfill(5)}.jpg")
            cv2.imwrite(img_path, frame[y:y+h, x:x+w])
            image_counter += 1
            print(f"✅  Saved: {img_path} ({image_counter}/{TOTAL_IMAGES_TO_CAPTURE})")
            cv2.waitKey(500) # หยุดรอ
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Dataset Collector - DEBUG MODE", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print(f"\n[INFO] เก็บข้อมูลเสร็จสิ้น! ได้รูปภาพทั้งหมด {image_counter} รูป")
cap.release()
cv2.destroyAllWindows()
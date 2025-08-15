# File: face_training.py (The Final, Safe & Unbiased Version)
import cv2
import os
import numpy as np
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
import time

# --- 1. การตั้งค่า ---
DATASET_PATH = 'dataset'
IMG_SIZE = 64
MIN_FACE_SIZE = 20
# จำนวนรูปใหม่ที่จะสร้างจาก 1 รูปต้นฉบับด้วย OpenCV
AUGMENTATIONS_PER_IMAGE = 5

# --- ฟังก์ชัน Augmentation ด้วย OpenCV (ปลอดภัย) ---
def augment_image(image):
    augmented_image = image.copy()
    
    # 1. พลิกภาพแนวนอน (สุ่ม 50%)
    if random.random() > 0.5:
        augmented_image = cv2.flip(augmented_image, 1)

    # 2. ปรับความสว่าง (สุ่ม)
    brightness_value = int(random.uniform(-40, 40))
    augmented_image = np.clip(augmented_image.astype(int) + brightness_value, 0, 255).astype(np.uint8)

    # 3. หมุนภาพเล็กน้อย (สุ่ม)
    rows, cols, _ = augmented_image.shape
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    augmented_image = cv2.warpAffine(augmented_image, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

    # 4. เพิ่ม Gaussian Blur เล็กน้อย (สุ่ม 50%)
    if random.random() > 0.5:
        augmented_image = cv2.GaussianBlur(augmented_image, (5, 5), 0)
        
    return augmented_image

# --- เริ่มต้น ---
print("="*50)
print("      เริ่มต้นกระบวนการฝึกโมเดล AI (Advanced)      ")
print("="*50)

print("\n[ขั้นตอนที่ 1/5] กำลังเตรียมและเพิ่มพูนข้อมูล (Augmentation)...")
faces = []
labels = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
person_folders = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]

for person_name in person_folders:
    person_dir = os.path.join(DATASET_PATH, person_name)
    print(f"  -> กำลังประมวลผลโฟลเดอร์: '{person_name}'")
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        image = cv2.imread(img_path)
        if image is None: continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))

        for (x, y, w, h) in detected_faces:
            face_roi = image[y:y+h, x:x+w]
            resized_face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            
            # --- เพิ่มข้อมูลต้นฉบับ 1 ครั้ง ---
            faces.append(resized_face)
            labels.append(person_name)
            
            # --- สร้างข้อมูลใหม่ด้วย Augmentation (OpenCV) ---
            for _ in range(AUGMENTATIONS_PER_IMAGE):
                augmented = augment_image(resized_face)
                faces.append(augmented)
                labels.append(person_name)
            
print(f"\n[ขั้นตอนที่ 1/5] ประมวลผลข้อมูลเสร็จสิ้น พบทั้งหมด {len(faces)} รูป (รวม Augment)")

# --- 2. แปลงข้อมูล ---
print("\n[ขั้นตอนที่ 2/5] กำลังแปลงข้อมูล (Preprocessing)...")
if len(faces) == 0: exit("❌ ไม่พบใบหน้า!")
if len(set(labels)) < 2: exit("❌ ต้องมีข้อมูลอย่างน้อย 2 คน")

faces = np.array(faces, dtype='float32') / 255.0
le = LabelEncoder()
labels_int = le.fit_transform(labels)
num_classes = len(le.classes_)
labels_onehot = to_categorical(labels_int, num_classes=num_classes)
(trainX, testX, trainY, testY) = train_test_split(faces, labels_onehot, test_size=0.20, stratify=labels_int, random_state=42)

# --- 3. คำนวณ Class Weights เพื่อลดความเอนเอียง ---
print("\n[ขั้นตอนที่ 3/5] กำลังคำนวณ Class Weights...")
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels_int),
    y=labels_int
)
class_weights_dict = dict(enumerate(class_weights))
print(f"  -> Class Weights คำนวณแล้ว: {class_weights_dict}")

# --- 4. สร้างและคอมไพล์โมเดล ---
print("\n[ขั้นตอนที่ 4/5] กำลังสร้างและคอมไพล์โมเดล...")
model = Sequential([
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # ===== ชั้นที่เพิ่มเข้ามา =====
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    # ==========================

    Flatten(),
    
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])
EPOCHS = 15
BS = 64 
opt = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# --- 5. ฝึกโมเดลพร้อม Class Weights ---
print("\n[ขั้นตอนที่ 5/5] เริ่มต้นการฝึกโมเดล...")
start_time = time.time()
model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), 
          epochs=EPOCHS, class_weight=class_weights_dict, verbose=1)
end_time = time.time()
print(f"\nการฝึกโมเดลเสร็จสิ้น ใช้เวลา: {end_time - start_time:.2f} วินาที")

# --- บันทึก (เหมือนเดิม) ---
(loss, acc) = model.evaluate(testX, testY, verbose=0)
print(f"\nผลการประเมินโมเดล:\n  -> Accuracy: {acc * 100:.2f}%")
model.save("face_recognition_model.h5")
with open("label_encoder.pickle", "wb") as f:
    f.write(pickle.dumps(le))
print("\n🎉🎉🎉 สร้าง 'สมอง' AI เวอร์ชันใหม่ที่ฉลาดขึ้นและเป็นกลางแล้ว! 🎉🎉🎉")
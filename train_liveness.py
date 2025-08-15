# File: train_liveness.py
import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
import time

# --- 1. การตั้งค่าและเตรียมข้อมูล ---
DATASET_PATH = 'liveness_dataset'
IMG_SIZE = 32 # สำหรับ Liveness ไม่ต้องใช้ภาพใหญ่มาก

print("="*50)
print("      เริ่มต้นกระบวนการฝึกโมเดล Liveness Detection      ")
print("="*50)
print(f"\n[ขั้นตอนที่ 1/5] กำลังอ่านข้อมูลจาก: '{DATASET_PATH}'...")

# ดึง path ของรูปภาพทั้งหมด
image_paths = list(paths.list_images(DATASET_PATH))
data = []
labels = []

# วนลูปในทุก path ของรูปภาพ
for image_path in image_paths:
    # ดึง "ฉลาก" (fake หรือ real) จากชื่อโฟลเดอร์
    label = image_path.split(os.path.sep)[-2]
    
    image = cv2.imread(image_path)
    if image is None: continue

    # ปรับขนาดภาพให้เท่ากัน
    resized_image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    data.append(resized_image)
    labels.append(label)

print(f"[ขั้นตอนที่ 1/5] ประมวลผลข้อมูลเสร็จสิ้น พบทั้งหมด {len(data)} รูปภาพ")

# --- 2. แปลงข้อมูล ---
print("\n[ขั้นตอนที่ 2/5] กำลังแปลงข้อมูล (Preprocessing)...")

data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels_int = le.fit_transform(labels)
labels_onehot = to_categorical(labels_int, 2) # มีแค่ 2 คลาส: fake, real

(trainX, testX, trainY, testY) = train_test_split(data, labels_onehot, test_size=0.25, random_state=42)

print(f"  -> ขนาดชุดข้อมูลสำหรับฝึก: {len(trainX)} รูป")
print(f"  -> ขนาดชุดข้อมูลสำหรับทดสอบ: {len(testX)} รูป")

# --- 3. สร้างโมเดล (LivenessNet) ---
print("\n[ขั้นตอนที่ 3/5] กำลังสร้างสถาปัตยกรรมของโมเดล (CNN)...")
# เราจะใช้โมเดลที่เล็กและเร็วกว่าสำหรับงานนี้
model = Sequential([
    Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(2, activation="softmax") # Output 2 ค่า (ความน่าจะเป็นของ fake และ real)
])
model.summary()

# --- 4. คอมไพล์และฝึกโมเดล ---
print("\n[ขั้นตอนที่ 4/5] กำลังคอมไพล์และเริ่มต้นการฝึกโมเดล...")
EPOCHS = 50
BS = 8 # ใช้ Batch Size เล็กๆ
opt = Adam(learning_rate=1e-4)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

start_time = time.time()
model.fit(trainX, trainY, batch_size=BS, validation_data=(testX, testY), epochs=EPOCHS, verbose=1)
end_time = time.time()
print(f"\n[ขั้นตอนที่ 4/5] การฝึกโมเดลเสร็จสิ้น ใช้เวลา: {end_time - start_time:.2f} วินาที")

# --- 5. ประเมินผลและบันทึก ---
print("\n[ขั้นตอนที่ 5/5] กำลังประเมินผลและบันทึกโมเดล...")
(loss, acc) = model.evaluate(testX, testY, verbose=0)
print(f"  -> ผลการประเมินโมเดล:\n     - Accuracy: {acc * 100:.2f}%")

print("  -> กำลังบันทึกโมเดลไปที่ 'liveness_model.h5'...")
model.save("liveness_model.h5")

print("  -> กำลังบันทึก Label Encoder ไปที่ 'liveness_label_encoder.pickle'...")
with open("liveness_label_encoder.pickle", "wb") as f:
    f.write(pickle.dumps(le))
    
print("\n🎉🎉🎉 สร้าง 'สมองนักจับผิด' สำเร็จแล้ว! 🎉🎉🎉")
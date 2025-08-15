# File: check_libraries.py
import sys
import os

print("="*60)
print("      🔬  ตรวจสอบความพร้อมของไลบรารีสำหรับโปรเจกต์  🔬")
print("="*60)

# ตรวจสอบเวอร์ชัน Python
print(f"\n🐍 กำลังใช้งาน Python เวอร์ชัน: {sys.version}")

# --- รายการไลบรารีที่จำเป็นสำหรับโปรเจกต์ทั้งหมด ---
# เราจะตรวจสอบไลบรารีหลักๆ ที่เราติดตั้งไป
required_libraries = {
    "tensorflow": "tensorflow",
    "keras": "keras",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "dlib": "dlib",
    "face-recognition": "face_recognition",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "Pillow": "PIL",
}

print("\n--- กำลังตรวจสอบไลบรารีที่จำเป็น ---")
all_good = True
for lib_name, import_name in required_libraries.items():
    try:
        # พยายาม import ไลบรารี
        lib = __import__(import_name)
        
        # ดึงเวอร์ชันออกมา (เป็นวิธีที่ปลอดภัยกว่า)
        version = "N/A"
        if hasattr(lib, '__version__'):
            version = lib.__version__
        # สำหรับ Keras เวอร์ชันใหม่อาจต้องใช้ keras.version()
        elif import_name == "keras" and hasattr(lib, 'version'):
            version = lib.version()

        print(f"  [ OK ] ✔️  {lib_name:<20} | เวอร์ชัน: {version}")
    except ImportError:
        print(f"  [FAIL] ❌  {lib_name:<20} | ไม่ได้ติดตั้ง! โปรดรัน: pip install {lib_name}")
        all_good = False
    except Exception as e:
        print(f"  [FAIL] ❌  {lib_name:<20} | เกิดข้อผิดพลาด: {e}")
        all_good = False

# --- ตรวจสอบไฟล์สำคัญอื่นๆ ---
print("\n--- กำลังตรวจสอบไฟล์สำคัญ ---")

# ตรวจสอบ Haar Cascade ของ OpenCV
try:
    import cv2
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        print("  [ OK ] ✔️  พบไฟล์ Haar Cascade สำหรับตรวจจับใบหน้า")
    else:
        print("  [FAIL] ❌  ไม่พบไฟล์ Haar Cascade! อาจมีปัญหากับการติดตั้ง OpenCV")
        all_good = False
except Exception as e:
    print(f"  [FAIL] ❌  ไม่สามารถตรวจสอบ Haar Cascade ได้: {e}")
    all_good = False


# --- สรุปผล ---
print("\n" + "-"*60)
if all_good:
    print("🎉 ยอดเยี่ยม! สภาพแวดล้อมของคุณพร้อมสำหรับโปรเจกต์ Face Recognition แล้ว 🎉")
else:
    print("⚠️ พบปัญหา! โปรดตรวจสอบข้อความ [FAIL] ❌ และทำการติดตั้งหรือแก้ไขตามคำแนะนำ")
print("="*60)
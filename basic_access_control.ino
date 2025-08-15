// File: basic_access_control.ino

// กำหนดขาที่ต่อกับ LED
const int redLedPin = 8;    // LED สีแดง (Unauthorized)
const int greenLedPin = 9;  // LED สีเขียว (Authorized)

void setup() {
  // เริ่มการสื่อสาร Serial ที่ Baud Rate 9600
  Serial.begin(9600);

  // ตั้งค่า Pin ของ LED เป็น OUTPUT
  pinMode(redLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);

  // เริ่มต้นด้วยสถานะ Unauthorized (ไฟแดงติด)
  digitalWrite(redLedPin, HIGH);
  digitalWrite(greenLedPin, LOW);
}

void loop() {
  // ตรวจสอบว่ามีข้อมูลส่งมาจาก Python หรือไม่
  if (Serial.available() > 0) {
    // อ่านข้อมูลเข้ามาทีละบรรทัด
    String command = Serial.readStringUntil('\n');
    command.trim(); // ตัดช่องว่างและอักขระขึ้นบรรทัดใหม่ออก

    if (command == "AUTHORIZED") {
      // ถ้าได้รับคำสั่ง "AUTHORIZED"
      // เปิดไฟเขียว, ปิดไฟแดง
      digitalWrite(greenLedPin, HIGH);
      digitalWrite(redLedPin, LOW);
    } else if (command == "UNAUTHORIZED") {
      // ถ้าได้รับคำสั่ง "UNAUTHORIZED" (หรือคำสั่งอื่น)
      // ปิดไฟเขียว, เปิดไฟแดง
      digitalWrite(greenLedPin, LOW);
      digitalWrite(redLedPin, HIGH);
    }
  }
}

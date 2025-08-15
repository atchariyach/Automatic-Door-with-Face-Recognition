# Automatic-Door-with-Face-Recognition
A Python-based face recognition system for controlling an automatic door, with liveness detection using a double-blink feature to prevent spoofing.
# 🔒 Ultimate Secure Face Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg) ![Keras](https://img.shields.io/badge/Keras-TensorFlow-red.svg) ![Arduino](https://img.shields.io/badge/Arduino-Control-cyan.svg)

A multi-layered, real-time face recognition system for secure access control. This project combines robust facial recognition with blink liveness detection to prevent spoofing attacks and integrates with Arduino for controlling physical devices like electronic locks or LEDs.

![Demo GIF](https://user-images.githubusercontent.com/username/repo/demo.gif)
*(Recommendation: Create a GIF showcasing the program in action and replace this placeholder link.)*

## ✨ Key Features

- **Real-time Face Recognition**: Accurately identifies authorized individuals from a live video stream.
- **Blink Liveness Detection**: A crucial security layer that requires users to blink, effectively preventing spoofing attempts using photos or videos.
- **Arduino Integration**: Sends `AUTHORIZED` and `UNAUTHORIZED` commands over a serial port to control external hardware (e.g., LEDs, relays, electric door locks).
- **Intelligent Dataset Creator**: A helper script that automates the creation of a high-quality image dataset. It only captures sharp, well-lit, and correctly posed facial images.
- **Modular & Clean Code**: The project is organized into distinct, easy-to-understand scripts for creating the dataset, training the model, and running the main application.

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Core Libraries**:
  - OpenCV: For real-time computer vision and image processing.
  - Dlib: For high-performance face detection and facial landmark prediction.
  - Keras (with TensorFlow backend): For building and training the Convolutional Neural Network (CNN) model.
  - Scikit-learn: For data preprocessing and model evaluation.
  - PySerial: For serial communication with the Arduino board.
- **Hardware (Optional)**:
  - Webcam
  - Arduino UNO (or a compatible board)

## 📂 Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── face_recognition_app.py   # <-- Run this file for the main application
├── create_dataset.py         # <-- Step 1: Create your image dataset
├── face_training.py          # <-- Step 2: Train the recognition model
└── arduino_code/
    └── basic_access_control.ino
```

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- [Python 3.9](https://www.python.org/downloads/) or newer installed.
- [Arduino IDE](https://www.arduino.cc/en/software) installed for uploading code to the Arduino board.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/ultimate-secure-face-recognition.git
cd ultimate-secure-face-recognition
```

### 3. Download Required Model Files

This project requires a pre-trained facial landmark predictor from dlib.
- Download **`shape_predictor_68_face_landmarks.dat`** from [dlib's official site](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
- Unzip the file and place the `shape_predictor_68_face_landmarks.dat` file in the root directory of the project.

### 4. Set Up a Python Environment & Install Dependencies

Using a virtual environment is highly recommended to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all required libraries from the requirements.txt file
pip install -r requirements.txt
```

### 5. Set Up the Arduino

1. Open the `arduino_code/basic_access_control.ino` file using the Arduino IDE.
2. Connect your Arduino board to your computer.
3. In the Arduino IDE, select the correct Board and Port from the `Tools` menu.
4. Click the **Upload** button.
5. Wire a red LED to Pin 8 and a green LED to Pin 9 (or modify the pins in the `.ino` file to match your setup).

---

## 📈 Workflow

Follow these three steps in order to get the system running.

### Step 1: Create the Dataset

Run the intelligent dataset creator script. This will capture high-quality images of each person you want the system to recognize.

```bash
python create_dataset.py
```
Follow the on-screen instructions (look straight, stay still) until the required number of images is captured. Repeat for each person.

### Step 2: Train the AI Model

Once the dataset is ready, train the face recognition model. This script will learn to distinguish between the individuals in your dataset.

```bash
python face_training.py
```
This process will take a few minutes. Upon completion, it will generate two crucial files: `face_recognition_model.h5` and `label_encoder.pickle`.

### Step 3: Run the Main Application

Before running, make sure to configure the correct Arduino serial port in the `face_recognition_app.py` file.

```python
# In face_recognition_app.py
ARDUINO_PORT = 'COM3' # <-- Change this to your Arduino's port (e.g., '/dev/ttyUSB0' on Linux)
```

Now, launch the main application:
```bash
python face_recognition_app.py
```
Look at the camera and **blink twice** to pass the liveness check. The system will then identify your face and send the appropriate signal to the Arduino.

## 📝 Notes

- The accuracy of the model is highly dependent on the quality of the dataset. Ensure good lighting and varied (but slight) head poses during dataset creation.
- If you consistently encounter "Bad Pose" or "Blurry" feedback, consider adjusting the `POSE_THRESHOLD` and `BLUR_THRESHOLD` values in the `create_dataset.py` script to better suit your camera and environment.


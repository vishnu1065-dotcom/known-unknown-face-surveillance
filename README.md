# 🔐 Known & Unknown Face Detection for Surveillance

An AI-based real-time surveillance system that uses **Raspberry Pi, USB Camera, Python, and OpenCV** to detect and recognize people. The system identifies registered individuals as **Known** and detects unregistered individuals as **Unknown**, making it suitable for basic intelligent surveillance applications.

## 🚀 Project Overview

Traditional surveillance systems continuously record video but may require a human operator to identify people manually.

This project adds an AI-based face recognition layer to the surveillance system.

The camera continuously monitors the surroundings and processes the captured video frames using computer vision.

The system can:

* 👤 Detect human faces in real time
* ✅ Recognize registered/known individuals
* ⚠️ Identify unknown individuals
* 📷 Capture surveillance frames
* 🖥️ Display detection results on the screen
* 🔔 Generate an alert when an unknown person is detected
* 🎥 Support integration with a web-based surveillance system

## 🎯 Objectives

The main objectives of this project are:

1. Develop a real-time face detection system.
2. Create a database of authorized/known faces.
3. Recognize registered individuals automatically.
4. Identify people who are not present in the registered database.
5. Provide an alert for unknown persons.
6. Build a foundation for an intelligent surveillance system.

## 🧰 Hardware Requirements

* Raspberry Pi 4 Model B
* USB Webcam
* MicroSD Card
* Raspberry Pi Power Supply
* Buzzer *(optional)*
* LED *(optional)*
* Wi-Fi connection

## 💻 Software Requirements

* Raspberry Pi OS
* Python 3
* OpenCV
* NumPy
* Face recognition / face encoding library
* VS Code / Thonny / Geany
* Linux Terminal

## 🧠 System Architecture

```text
              ┌─────────────────┐
              │    USB Camera   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Video Capture  │
              │    OpenCV       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Face Detection │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Face Recognition│
              └────────┬────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      ┌─────────────┐     ┌─────────────┐
      │    Known    │     │   Unknown   │
      │    Person   │     │    Person   │
      └──────┬──────┘     └──────┬──────┘
             │                   │
             ▼                   ▼
       Display Name        Alert / Capture
```

## ⚙️ Working Principle

### 1. Camera Initialization

The USB webcam is connected to the Raspberry Pi and initialized using OpenCV.

The camera continuously captures video frames.

### 2. Face Detection

Each frame is processed to locate human faces.

When a face is detected, the system creates a region around the detected face.

### 3. Face Encoding

The detected face is converted into a numerical representation called a **face encoding**.

The encoding contains features that can be compared with previously registered faces.

### 4. Known Face Database

Images of authorized people are stored in a database.

Example:

```text
known_faces/
│
├── person_1/
│   └── image.jpg
│
├── person_2/
│   └── image.jpg
│
└── person_3/
    └── image.jpg
```

The system generates face encodings from these images.

### 5. Face Matching

The live camera face encoding is compared with the stored face encodings.

If a sufficiently close match is found:

```text
Person Detected
      ↓
Face Matching
      ↓
Match Found
      ↓
KNOWN PERSON
```

If no sufficiently close match is found:

```text
Person Detected
      ↓
Face Matching
      ↓
No Match
      ↓
UNKNOWN PERSON
```

### 6. Unknown Person Alert

When an unknown person is detected, the system can:

* Display `UNKNOWN PERSON`
* Capture an image
* Save the detection time
* Activate a buzzer
* Send an alert to a web server
* Store the event for later review

## 📁 Project Structure

```text
known-unknown-face-surveillance/
│
├── known_faces/
│   ├── person_1/
│   ├── person_2/
│   └── person_3/
│
├── unknown_faces/
│
├── captured_images/
│
├── src/
│   ├── face_detection.py
│   ├── face_recognition.py
│   └── surveillance.py
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

## 🔄 System Workflow

```text
Start
  │
  ▼
Initialize Camera
  │
  ▼
Capture Video Frame
  │
  ▼
Detect Face
  │
  ├── No Face ──────► Continue Monitoring
  │
  ▼
Extract Face Features
  │
  ▼
Compare With Database
  │
  ├── Match ────────► Known Person
  │                     │
  │                     ▼
  │                 Display Name
  │
  └── No Match ─────► Unknown Person
                        │
                        ▼
                    Save Image
                        │
                        ▼
                    Trigger Alert
```

## 🖥️ Expected Output

### Known Person

```text
┌─────────────────────────────┐
│                             │
│       ┌────────────┐        │
│       │            │        │
│       │    FACE    │        │
│       │            │        │
│       └────────────┘        │
│                             │
│       Name: Vishnu          │
│       Status: KNOWN         │
│                             │
└─────────────────────────────┘
```

### Unknown Person

```text
┌─────────────────────────────┐
│                             │
│       ┌────────────┐        │
│       │            │        │
│       │    FACE    │        │
│       │            │        │
│       └────────────┘        │
│                             │
│       Status: UNKNOWN       │
│       ⚠ ALERT GENERATED     │
│                             │
└─────────────────────────────┘
```

## 🔔 Surveillance Alert

The system can be extended to activate an alert whenever an unknown person is detected.

Example:

```text
Unknown Person Detected
          ↓
Capture Image
          ↓
Save Detection Time
          ↓
Activate Buzzer
          ↓
Send Web Alert
```

## 🌐 Future Web Integration

This project can be integrated with a web-based surveillance dashboard.

The dashboard can display:

* 📹 Live camera feed
* 👤 Current detected person
* ✅ Known/Unknown status
* 🕒 Detection time
* 📸 Captured unknown-person images
* 🚨 Alert status
* 📊 Detection history

Example:

```text
========================================
       AI SURVEILLANCE DASHBOARD
========================================

Camera Status       : ONLINE
Person Detected     : YES
Identity            : UNKNOWN
Alert Status        : ACTIVE

----------------------------------------
           LIVE CAMERA FEED
----------------------------------------

        [ Camera Stream ]

----------------------------------------
Recent Detection:
Unknown Person
20-09-2026  01:45 AM
----------------------------------------
```

## 🛠️ Technologies Used

| Technology       | Purpose                 |
| ---------------- | ----------------------- |
| Raspberry Pi 4   | Main processing unit    |
| USB Webcam       | Live video acquisition  |
| Python           | Application development |
| OpenCV           | Computer vision         |
| NumPy            | Numerical processing    |
| Face Recognition | Face identification     |
| Wi-Fi            | Network communication   |
| Web Server       | Remote surveillance     |

## 📌 Applications

This system can be used as a foundation for:

* Smart surveillance
* Restricted-area monitoring
* Laboratory security
* Office monitoring
* Home security
* Smart door access systems
* Robotics surveillance
* AI security systems

## 🚀 Future Improvements

The project can be further improved by adding:

* Real-time web dashboard
* Email/Telegram notifications
* Face recognition database management
* Automatic image logging
* Multiple-person recognition
* Night-time surveillance
* Cloud storage
* Mobile notifications
* PTZ camera control
* Integration with an autonomous surveillance rover

## 🔐 Privacy & Security

Face images and biometric data are sensitive. Use this project only with appropriate authorization and consent, and store collected data securely. Avoid using it to monitor people without their knowledge or a legitimate purpose.

## 👨‍💻 Author

**Vishnu M.**

Embedded Systems | IoT | Robotics | Computer Vision | AI

## ⭐ Project Highlights

```text
✔ Real-Time Face Detection
✔ Known Person Recognition
✔ Unknown Person Detection
✔ Raspberry Pi Based
✔ OpenCV Computer Vision
✔ Surveillance Alert System
✔ Expandable Web Dashboard
✔ Robotics Integration Ready
```

## 📜 License

This project is intended for educational, research, and authorized surveillance applications.

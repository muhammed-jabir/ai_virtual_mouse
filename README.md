🖐️👁️ AI-Powered Virtual Mouse

Control your computer using hand gestures and eye movements with Python, OpenCV, MediaPipe, and PyAutoGUI.

An AI-powered touchless mouse system that uses a webcam to track hand landmarks, gestures, face landmarks, and iris movement and converts them into real-time mouse actions.

🚀 Features
🖐️ Hand Control
Real-time hand tracking
Index finger cursor movement
Cursor smoothing
Dead-zone stabilization
🤏 Thumb + Index pinch → Left Click
✌️ Index + Middle fingers → Right Click
🖐️ Hand movement → Scrolling
👁️ Eye Control
Face Mesh tracking
Iris detection
Left / Center / Right gaze detection
Eye-based cursor control
Gaze smoothing and ratio calculation
🔥 Hybrid Control
Eye + finger cursor control
Keyboard-controlled modes
Safe default hand mode
Emergency ESC exit
🧠 How It Works
             WEBCAM
                │
                ▼
             OpenCV
                │
        ┌───────┴───────┐
        ▼               ▼
   MediaPipe         MediaPipe
     Hands           Face Mesh
        │               │
        ▼               ▼
 Hand Landmarks     Iris Landmarks
        │               │
        ▼               ▼
 Gesture Engine     Gaze Engine
        │               │
        └───────┬───────┘
                ▼
          MODE CONTROLLER
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
     HAND      EYE     HYBRID
       │        │        │
       └────────┼────────┘
                ▼
        MouseController
                │
                ▼
            PyAutoGUI
                │
                ▼
          OS Mouse Cursor
☝️ How Finger Cursor Control Works

MediaPipe provides 21 hand landmarks.

The index fingertip is landmark 8.

index_tip = hand_landmarks.landmark[8]


index_x = int(
    index_tip.x * frame_width
)


index_y = int(
    index_tip.y * frame_height
)

The camera coordinates are then converted into screen coordinates:

Camera Coordinates
       ↓
Coordinate Mapping
       ↓
Screen Coordinates
       ↓
PyAutoGUI
       ↓
Mouse Cursor

The cursor is stabilized using smoothing and a dead zone to reduce hand-tracking jitter.

🤏 Left Click

The system measures the distance between:

Thumb Tip  → Landmark 4
Index Tip  → Landmark 8
pinch_distance = calculate_distance(
    thumb_x,
    thumb_y,
    index_x,
    index_y
)

When the distance becomes smaller than the configured threshold:

Thumb + Index
     ↓
   Pinch
     ↓
 Left Click

A release threshold prevents repeated clicks while holding the pinch.

✌️ Right Click

Right click uses this gesture:

Index  → Extended
Middle → Extended
Ring   → Folded
Pinky  → Folded
      ☝️
      🖕
      │
      └── Right Click

A cooldown prevents multiple right-clicks from being triggered continuously.

🖐️ Scrolling

Vertical hand movement is converted into scroll input.

Hand Up
  ↓
Scroll Up


Hand Down
  ↓
Scroll Down

The movement is calculated from the change in finger position and passed to:

mouse.scroll(amount)
👁️ Eye Tracking

MediaPipe Face Mesh provides iris landmarks.

Left Iris  → 468–472
Right Iris → 473–477

The project calculates the iris position relative to the eye boundaries.

Iris Position
     ↓
Gaze Ratio
     ↓
LEFT / CENTER / RIGHT

Example:

0.0        0.5        1.0
│----------│----------│
LEFT      CENTER     RIGHT

Eye tracking can be calibrated further for better accuracy.

🔀 Control Modes

The project supports three cursor modes.

Key	Mode	Description
F8	🖐️ Hand	Finger controls cursor
F7	👁️ Eye	Eyes control cursor
F6	🔥 Hybrid	Eye + finger control
ESC	🛑 Exit	Stop application
🖐️ F8 — Hand Mode
Index Finger → Cursor
Pinch        → Left Click
Two Fingers  → Right Click
Hand Motion  → Scroll
👁️ F7 — Eye Mode
Eye Movement
     ↓
Gaze Detection
     ↓
Cursor Movement
🔥 F6 — Hybrid Mode

Both eye and finger tracking are active.

The eye provides broader cursor positioning while finger tracking helps with fine control.

Conceptually:

final_x = eye_x * 0.7 + finger_x * 0.3
final_y = eye_y * 0.7 + finger_y * 0.3

The weights can be adjusted in future versions.

🛡️ Safety

The project starts in Hand Mode instead of automatically enabling eye control.

If the cursor behaves unexpectedly:

Press ESC

The application will stop and release the webcam.

cap.release()
cv2.destroyAllWindows()
📁 Project Structure
ai_virtual_mouse/
│
├── main.py
├── config.py
├── eye_test.py
├── requirements.txt
├── README.md
│
├── hand_tracking/
│   ├── __init__.py
│   └── hand_detector.py
│
├── eye_tracking/
│   ├── __init__.py
│   └── eye_tracker.py
│
└── mouse_control/
    ├── __init__.py
    └── mouse_controller.py
🧩 Main Components
main.py

Controls the complete application:

Webcam
Hand tracking
Eye tracking
Gesture recognition
Scrolling
Cursor control
Mode switching
config.py

Stores configurable settings:

Camera resolution
Detection confidence
Cursor smoothing
Dead zone
Pinch threshold
Scroll sensitivity
Eye settings
hand_detector.py

Handles MediaPipe hand detection and landmarks.

eye_tracker.py

Handles Face Mesh, iris tracking, and gaze detection.

mouse_controller.py

Provides a simple interface for:

mouse.move()
mouse.left_click()
mouse.right_click()
mouse.scroll()

using PyAutoGUI.

🛠️ Tech Stack
Technology	Purpose
🐍 Python	Core development
👁️ MediaPipe	Hand & face tracking
📷 OpenCV	Webcam & image processing
🖱️ PyAutoGUI	Mouse automation
🔢 NumPy	Numerical processing
🤖 Computer Vision	Gesture & gaze recognition
⚙️ Installation
1. Clone
git clone https://github.com/YOUR_USERNAME/ai_virtual_mouse.git
cd ai_virtual_mouse
2. Create virtual environment
python -m venv venv
3. Activate

Windows:

venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt

Or:

pip install opencv-python mediapipe pyautogui numpy
▶️ Run
python main.py

For eye-tracking testing:

python eye_test.py
🎮 Controls
Index Finger              → Move Cursor
Thumb + Index Pinch       → Left Click
Index + Middle Fingers    → Right Click
Hand Vertical Movement    → Scroll


F8                        → Hand Mode
F7                        → Eye Mode
F6                        → Hybrid Mode
ESC                       → Exit
✅ Current Status
 Webcam integration
 MediaPipe hand tracking
 Index finger cursor
 Cursor smoothing
 Dead-zone stabilization
 Pinch left-click
 Right-click gesture
 Gesture scrolling
 Face Mesh
 Iris detection
 Gaze detection
 Eye cursor architecture
 Hand mode
 Eye mode
 Hybrid mode
 Keyboard mode switching
 Emergency exit
🚧 Future Improvements
 Drag & Drop gesture
 Double-click gesture
 Middle click
 Better scrolling
 Blink detection
 Eye-based clicking
 Dwell clicking
 Automatic eye calibration
 ML-based gesture recognition
 Custom gesture training
 Desktop GUI
 User-configurable sensitivity
🌍 Possible Applications
♿ Accessibility
🤖 Robotics
🥽 AR / VR
🖥️ Smart Displays
🏭 Industrial interfaces
🏥 Touchless healthcare interfaces
🏪 Interactive kiosks
🧑‍💻 Human-Computer Interaction research
🧠 What This Project Demonstrates

This project combines:

Computer Vision
      +
Hand Tracking
      +
Eye Tracking
      +
Gesture Recognition
      +
Coordinate Mapping
      +
Mouse Automation
      ↓
Touchless Human-Computer Interaction

The goal is to explore how AI and computer vision can replace traditional input devices with natural human gestures and eye movement.

👨‍💻 Author

Muhammed Jabir M T

Odoo Developer | Python Developer | Web Developer

Interested in:

AI · Computer Vision · Python · Robotics · Automation · HCI

⭐ Support

If you like the project:

⭐ Star the repository
🍴 Fork it
🐛 Report an issue
💡 Suggest an improvement

🚀 Project Vision

Turn human movement into a natural computer interface.

Built with ❤️ using Python + OpenCV + MediaPipe + PyAutoGUI.

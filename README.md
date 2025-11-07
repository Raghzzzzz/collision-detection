# 🚦 Traffic AI Master – Intelligent Traffic Surveillance System

This project is a **multi-module traffic analysis system** powered by **YOLOv8 (Ultralytics)** and **DeepSORT** for real-time vehicle tracking and behavioral analytics.  
It includes five AI-powered features for automated traffic monitoring, anomaly detection, and violation alerting.

---

## 🧠 Features

### 1️⃣ Vehicle Speed & Violation Detection
- Detects and tracks vehicles using YOLOv8 + DeepSORT.
- Calculates real-time speed based on pixel distance and frame rate.
- Highlights vehicles exceeding the speed limit (>45 km/h).
- Saves annotated output video with detected speeds and violations.

---

### 2️⃣ Lane Detection + Traffic Density Monitoring
- Uses **Canny edge detection** and **Hough Transform** to estimate lanes.
- Detects vehicles and counts them per lane using YOLOv8 + DeepSORT.
- Calculates traffic density dynamically.
- Displays warnings for high-density traffic conditions.

---

### 3️⃣ Wrong-Way Detection
- Tracks direction of vehicle motion.
- Alerts when vehicles move **opposite** to the expected flow (user-input: `in` / `out`).
- Highlights wrong-way vehicles in **red** and correct vehicles in **green**.

---

### 4️⃣ Collision Detection (YOLO + IoU)
- Detects possible collisions based on **bounding box overlap (IoU)** between vehicles.
- Marks collisions with red boxes and displays **“Collision Happened”** alert.
- Simple, fast, and accurate IoU-based logic — **no tracking dependency**.

---

### 5️⃣ Anomaly Detection
- Detects **sudden stops** and **zigzag driving** patterns using DeepSORT tracking and motion analysis.
- Computes vehicle speed using temporal movement smoothing.
- Alerts and labels detected anomalies in real-time.

---
## 🧩 System Architecture

┌──────────────────────────┐
│ Input Video │
└─────────────┬────────────┘
│
YOLOv8 Object Detection
│
DeepSORT Multi-Tracking
│
┌────────┼─────────────┐
│ │ │
Speed Lane Density Anomaly
│ │ │
▼ ▼ ▼
Collision Wrong-Way Output Videos
Detection Detection
---

## ⚙️ Requirements

### 🧰 Dependencies
Install these before running:
```bash
pip install ultralytics
pip install opencv-python
pip install torch torchvision torchaudio
pip install numpy
pip install deep-sort-realtime
💻 Hardware
GPU (CUDA) recommended for real-time performance.

Works on CPU (slower but functional).

🗂️ Project Structure
Traffic-AI-Master/
│
├── traffic_ai_master.py     # Main script with 5 modules
├── README.md                # Documentation file
├── output_speed_violation.mp4
├── output_lane_density.mp4
├── output_wrongway.mp4
├── output_anomaly.mp4
├── output_collision.mp4
└── input_videos/
    ├── input_trafic4.mp4
    ├── WhatsApp Video 2025-11-06...
    └── WhatsApp Video 2025-11-07...


🚀 How to Run

Clone or copy this repository to your local system.

Place your input videos in the project folder and update the paths inside the script if needed.

Run the program:
        python traffic_ai_master.py

Choose a mode:
        1️⃣ Vehicle Speed + Violation Detection
        2️⃣ Lane Detection + Traffic Density
        3️⃣ Wrong-Way Detection
        4️⃣ Collision Detection
        5️⃣ Anomaly Detection

Press ‘q’ to quit any mode.

Processed videos are automatically saved as:

output_speed_violation.mp4

output_lane_density.mp4

output_wrongway.mp4

output_anomaly.mp4

🧑‍💻 Author

Raghav
🎓 Chennai Institute of Technology
💼 Pursuing Cyber Security (2nd Year)
💡 Interests: Cybersecurity, AI, Web Development, and CTF Challenges

🏁 Output Previews
Module	Output
Speed & Violation	🚗 Speed overlay + Violation alert
Lane Density	🛣️ Lane count + Density value
Wrong-Way	🔄 Direction check alert
Collision	💥 Collision marking in red
Anomaly	⚠️ ZigZag & Sudden Stop alerts


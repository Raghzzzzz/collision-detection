🚗 Vehicle Collision Detection using YOLOv8 (README)
📘 Overview

This project performs real-time vehicle collision detection using:

YOLOv8m for detecting vehicles

IoU calculation to detect overlapping bounding boxes

OpenCV for video processing and visualization

When two vehicles overlap beyond a specified IoU threshold, the system highlights them in red and displays a “Collision Happened” alert.



🔍 How It Works
1. Vehicle Detection

Using YOLOv8m, the following COCO vehicle classes are detected:

Car (ID: 2)

Motorcycle (ID: 3)

Bus (ID: 5)

Truck (ID: 7)

Bounding boxes are drawn in green.

2. Collision Detection (IoU-Based)

The code computes Intersection over Union (IoU) between every pair of detected vehicles.

If:     
       IoU > threshold  (default = 0.1)


Then both boxes are marked as:

Red bounding box

A label: "Collision Happened"

This is a simple but effective approach for detecting physical overlap or near-touching vehicles in traffic videos.

collision_detection/
│
├── collision_detection.py     # Main detection script
├── video.mp4                  # Input video
└── README.md                  # Documentation

🛠 Requirements

Install dependencies:

pip install ultralytics
pip install opencv-python
pip install numpy

YOLOv8m will download automatically on the first run.


▶️ Usage
1. Set the video path:
          video_path = r"C:\path\to\your\video.mp4"

2. Run the script:
          python collision_detection.py

3. Controls

Press Q to exit the video window.

📌 Code Breakdown
IoU Function

Calculates overlap between two bounding boxes.

detect_vehicle_collisions()

Iterates over all vehicle boxes and checks if IoU > threshold.

YOLO Detection

Extracts bounding boxes and filters by COCO vehicle class IDs.

Visualization

Green → Normal vehicles

Red → Vehicles involved in collision

Text overlay → “Collision Happened”

🎯 Example Output

✔ Vehicles detected in green
✔ Colliding vehicles highlighted in red
✔ Warning text above collision area
✔ Real-time display using OpenCV

⚠️ Limitations

IoU-based collision is purely bounding-box overlap, not physical crash detection.

Works best with fixed cameras and clear visibility.

May detect false collisions in crowded scenes.   

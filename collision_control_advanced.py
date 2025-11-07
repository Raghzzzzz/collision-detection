from ultralytics import YOLO
import cv2
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8m.pt')

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def detect_vehicle_collisions(vehicle_boxes, threshold=0.1):
    collision_pairs = []
    for i in range(len(vehicle_boxes)):
        for j in range(i + 1, len(vehicle_boxes)):
            if iou(vehicle_boxes[i], vehicle_boxes[j]) > threshold:
                collision_pairs.append((i, j))
    return collision_pairs

video_path = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\WhatsApp Video 2025-11-07 at 11.28.38_d175f92f.mp4"
cap = cv2.VideoCapture(video_path)

vehicle_classes = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    vehicle_boxes = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            if int(cls) in vehicle_classes:
                vehicle_boxes.append(box)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

    collisions = detect_vehicle_collisions(vehicle_boxes)
    for (i, j) in collisions:
        box1 = vehicle_boxes[i]
        box2 = vehicle_boxes[j]
        cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 0, 255), 3)
        cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 0, 255), 3)
        x_text = int(min(box1[0], box2[0]))
        y_text = int(min(box1[1], box2[1])) - 10
        cv2.putText(frame, 'Collision Happened', (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Vehicle Collision Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

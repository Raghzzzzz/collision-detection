import os
import cv2
import time
import math
import torch
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================================
# CONFIGURATION
# ==========================================================
VIDEO_PATH = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\WhatsApp Video 2025-11-06 at 10.40.58_6e35b62e.mp4"

# ==========================================================
# COMMON HELPERS
# ==========================================================
def cuda_check(model):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.to("cuda")
        print("✅ CUDA available:", torch.cuda.get_device_name(0))
    else:
        print("❌ CUDA not available. Running on CPU.")
    return use_cuda


# ==========================================================
# 1️⃣ VEHICLE SPEED & VIOLATION DETECTION
# ==========================================================
def vehicle_speed_violation():
    print("\n🚗 Running: Speed Detection & Violation Checker\n")

    VIDEO_PATH = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\WhatsApp Video 2025-11-06 at 10.40.58_6e35b62e.mp4"
    model = YOLO("yolov8m.pt")
    cuda_check(model)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("output_speed_violation.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 384))

    y_history, centers, times = {}, {}, {}
    custom_id_counter, custom_ids, assigned_ids = 1, {}, set()
    st = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 384))
        results = list(model(frame, imgsz=320))[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2, y2], float(box.conf[0]), cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        now = time.time()

        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2
            y_history.setdefault(tid, []).append(cy)
            y_history[tid] = y_history[tid][-3:]

            moving_toward = len(y_history[tid]) == 3 and y_history[tid][0] < y_history[tid][1] < y_history[tid][2]
            if moving_toward:
                if tid not in custom_ids:
                    while custom_id_counter in assigned_ids:
                        custom_id_counter += 1
                    custom_ids[tid] = custom_id_counter
                    assigned_ids.add(custom_id_counter)
                    custom_id_counter += 1

                if tid in centers:
                    dx, dy = cx - centers[tid][0], cy - centers[tid][1]
                    pix_dist = math.hypot(dx, dy)
                    dt = now - times.get(tid, now)
                    if dt <= 0:
                        dt = 1 / fps
                    speed = (pix_dist * 0.05 / dt) * 3.6 + 20
                    color = (0, 0, 255) if speed > 45 else (0, 255, 0)
                    cv2.putText(frame, f"{speed:.1f} km/h", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if speed > 45:
                        cv2.putText(frame, "Violation!", (l, b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                centers[tid], times[tid] = (cx, cy), now
                cv2.rectangle(frame, (l, t), (r - (r // 2), b - (b // 2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{custom_ids[tid]}", (l, b + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Speed Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print("\n✅ Saved: output_speed_violation.mp4")
    print(f"⏱ Time: {time.time() - st:.1f} sec")


# ==========================================================
# 2️⃣ LANE DETECTION + TRAFFIC DENSITY
# ==========================================================
def lane_density_monitor():
    print("\n🛣️ Running: Lane Detection + Traffic Density\n")
    model = YOLO("yolov8m.pt")
    cuda_check(model)
    tracker = DeepSort(max_age=30)

    def estimate_lanes(frame):
        gray = cv2.cvtColor(cv2.resize(frame, (640, 384)), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
        mask = np.zeros_like(edges)
        pts = np.array([[50, 384], [280, 250], [360, 250], [590, 384]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        roi = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)
        if lines is None:
            return 3
        xs = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if y2 < y1:
                xs += [x1, x2]
        xs.sort()
        c = []
        for x in xs:
            if not c or min(abs(x - cc) for cc in c) > 30:
                c.append(x)
        return len(c)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("output_lane_density.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 384))
    ret, frame = cap.read()
    lanes = estimate_lanes(frame) if ret else 3
    y_history = {}
    ZONE_Y = 192

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 384))
        results = list(model(frame, imgsz=320))[0]

        dets = []
        for b in results.boxes:
            c = int(b.cls[0])
            if c in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append(([x1, y1, x2, y2], float(b.conf[0]), c))
        tracks = tracker.update_tracks(dets, frame=frame)
        zone = 0
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cy = (t + b) // 2
            y_history.setdefault(tid, []).append(cy)
            y_history[tid] = y_history[tid][-3:]
            moving = len(y_history[tid]) == 3 and y_history[tid][0] < y_history[tid][1] < y_history[tid][2]
            if moving and cy > ZONE_Y:
                zone += 1
            cv2.rectangle(frame, (l, t), (r-(r//2), b-(b//2)), (0, 255, 0), 2)
        density = zone / lanes
        cv2.putText(frame, f"Lanes:{lanes} Vehicles:{zone} Density:{density:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if density >= 5.0:
            cv2.putText(frame, "⚠️ TRAFFIC WARNING ⚠️", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        out.write(frame)
        cv2.imshow("Lane Density", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    print("\n✅ Saved: output_lane_density.mp4")


# ==========================================================
# 3️⃣ WRONG WAY DETECTION
# ==========================================================
def wrong_way_detection():
    print("\n🔄 Running: Wrong-Way Detection\n")
    
    # Ask user for expected direction
    expected_direction = input("Enter expected direction ('in' or 'out'): ").strip().lower()
    if expected_direction not in ["in", "out"]:
        print("❌ Invalid input. Please enter 'in' or 'out'.")
        return

    VIDEO_PATH = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\WhatsApp Video 2025-11-07 at 11.50.07_b410c9fd.mp4"
    model = YOLO("yolov8m.pt")
    cuda_check(model)
    tracker = DeepSort(max_age=30)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("output_wrongway.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 384))
    y_hist = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 384))
        res = list(model(frame, imgsz=320))[0]
        dets = []
        for b in res.boxes:
            c = int(b.cls[0])
            if c in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dets.append(([x1, y1, x2, y2], float(b.conf[0]), c))
        tracks = tracker.update_tracks(dets, frame=frame)
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cy = (t + b) // 2
            y_hist.setdefault(tid, []).append(cy)
            y_hist[tid] = y_hist[tid][-3:]
            hist = y_hist[tid]
            toward = len(hist) == 3 and hist[0] < hist[1] < hist[2]
            away = len(hist) == 3 and hist[0] > hist[1] > hist[2]
            wrong = (expected_direction == "in" and away) or (expected_direction == "out" and toward)
            color = (0, 0, 255) if wrong else (0, 255, 0)
            label = f"ID {tid}" + (" - Wrong Way" if wrong else "")
            cv2.rectangle(frame, (l, t), (r - (r // 2), b - (b // 2)), color, 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        out.write(frame)
        cv2.imshow("Wrong Way", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    print("\n✅ Saved: output_wrongway.mp4")


# ==========================================================
# 4️⃣ COLLISION DETECTION
# ==========================================================
def collision_detection():
    from ultralytics import YOLO
    import cv2
    import numpy as np

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
        if union_area == 0:
            return 0
        return inter_area / union_area

    def detect_vehicle_collisions(vehicle_boxes, threshold=0.1):
        collision_pairs = []
        for i in range(len(vehicle_boxes)):
            for j in range(i + 1, len(vehicle_boxes)):
                if iou(vehicle_boxes[i], vehicle_boxes[j]) > threshold:
                    collision_pairs.append((i, j))
        return collision_pairs

    video_path = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\input_trafic4.mp4"
    cap = cv2.VideoCapture(video_path)
    vehicle_classes = [2, 3, 5, 7]  # car, motorbike, bus, truck

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
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

# ==========================================================
# 5️⃣ ANOMALY DETECTION
# ==========================================================
def anomaly_detection():
    print("\n⚠ Running: Anomaly Detection (No IDs)")

    # ✅ Separate video path only for anomaly mode
    ANOMALY_VIDEO = r"C:\Users\ragha\OneDrive\Desktop\traffic detection\traffic detection\WhatsApp Video 2025-11-06 at 10.40.58_6e35b62e.mp4"

    if not os.path.exists(ANOMALY_VIDEO):
        print("❌ File not found:", ANOMALY_VIDEO)
        return

    model = YOLO("yolov8m.pt")
    cuda_check(model)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(ANOMALY_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "output_anomaly.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    # --- Tracking memory ----
    centers = {}
    last_t = {}
    smooth_speed = {}
    x_hist = {}
    speed_hist = {}

    # --- Settings ---
    SHRINK = 0.45
    EMA_ALPHA = 0.3
    SPEED_OFFSET = 40

    st = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        result = list(model(frame, imgsz=640))[0]

        dets = []
        for b in result.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])

            if cls not in [2,3,5,7]:  # vehicle classes
                continue
            if conf < 0.30:
                continue

            x1, y1, x2, y2 = map(int, b.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            # remove tiny noise
            if w*h < 600:
                continue

            dets.append(([x1, y1, w, h], conf, cls))

        tracks = tracker.update_tracks(dets, frame=frame)

        for tr in tracks:
            if not tr.is_confirmed():
                continue

            tid = tr.track_id
            l, t, r, b = map(int, tr.to_ltrb())
            cx = (l+r)//2
            cy = (t+b)//2

            # store center history
            x_hist.setdefault(tid, []).append(cx)
            if len(x_hist[tid]) > 20:
                x_hist[tid] = x_hist[tid][-20:]

            # -----------------------
            # SPEED
            # -----------------------
            if tid in centers:
                px, py = centers[tid]
                dist = math.hypot(cx - px, cy - py)
                dt = now - last_t.get(tid, now)
                if dt <= 0:
                    dt = 1 / fps
                raw_speed = (dist * 0.05 / dt) * 3.6
            else:
                raw_speed = 0

            prev = smooth_speed.get(tid, raw_speed)
            smoothed = prev + EMA_ALPHA * (raw_speed - prev)
            smooth_speed[tid] = smoothed
            speed_display = smoothed + SPEED_OFFSET

            speed_hist.setdefault(tid, []).append(smoothed)
            if len(speed_hist[tid]) > 10:
                speed_hist[tid] = speed_hist[tid][-10:]

            centers[tid] = (cx, cy)
            last_t[tid] = now

            # -----------------------
            # ANOMALY CHECK
            # -----------------------
            anomalies = []

            # Sudden stop
            if len(speed_hist[tid]) >= 6:
                sp = speed_hist[tid][-6:]
                prev_avg = sum(sp[:3]) / 3
                curr_avg = sum(sp[3:]) / 3

                if prev_avg > 20 and curr_avg < 3 and (prev_avg - curr_avg) > 15:
                    anomalies.append("Sudden Stop")

            # ZigZag
            if len(x_hist[tid]) >= 15:
                xs = x_hist[tid][-15:]
                dx = [xs[i] - xs[i-1] for i in range(1, len(xs))]
                dx_clean = [d for d in dx if abs(d) > 2]

                flips = 0
                for i in range(1, len(dx_clean)):
                    if dx_clean[i] * dx_clean[i-1] < 0:
                        flips += 1

                if flips >= 2:
                    anomalies.append("ZigZag")

            # -----------------------
            # SMALL CENTERED BOX
            # -----------------------
            w2 = r - l
            h2 = b - t
            new_w = int(w2 * (1 - SHRINK))
            new_h = int(h2 * (1 - SHRINK))
            l2 = cx - new_w // 2
            r2 = cx + new_w // 2
            t2 = cy - new_h // 2
            b2 = cy + new_h // 2

            color = (0, 0, 255) if anomalies else (0, 255, 0)
            cv2.rectangle(frame, (l2, t2), (r2, b2), color, 2)

            # Speed label
            cv2.putText(frame, f"{speed_display:.1f} km/h",
                        (l2, t2-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0,255,255),
                        2)

            # Anomaly labels
            y_text = t2 - 30
            for an in anomalies:
                cv2.putText(frame, an,
                            (l2, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0,0,255),
                            2)
                y_text -= 20

        out.write(frame)
        cv2.imshow("Anomaly Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n✅ Saved: output_anomaly.mp4")


# ==========================================================
# 🎯 MAIN MENU
# ==========================================================
if __name__ == "__main__":
    print("\n================= TRAFFIC AI MASTER =================")
    print("Video:", VIDEO_PATH)
    print("Choose a mode to run:")
    print("1️⃣  Vehicle Speed + Violation Detection")
    print("2️⃣  Lane Detection + Traffic Density")
    print("3️⃣  Wrong-Way Detection")
    print("4️⃣  Collision Prediction & Detection")
    print("5️⃣  Anomaly Detection (ZigZag / Sudden Stop )")
    print("=====================================================")

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        vehicle_speed_violation()
    elif choice == "2":
        lane_density_monitor()
    elif choice == "3":
        wrong_way_detection()
    elif choice == "4":
        collision_detection()
    elif choice == "5":
        anomaly_detection()
    else:
        print("❌ Invalid choice. Exiting.")
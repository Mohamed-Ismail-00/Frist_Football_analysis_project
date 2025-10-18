from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

frames_folder = "frames"
output_folder = "ball_tracking"
os.makedirs(output_folder, exist_ok=True)

for frame_file in sorted(os.listdir(frames_folder)):
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    results = model(frame)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name == "sports ball":  # نركز على الكورة فقط
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ball {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    output_path = os.path.join(output_folder, frame_file)
    cv2.imwrite(output_path, frame)

print("✅ تم تحديد الكره فقط وحفظ الفريمات في 'ball_tracking'")

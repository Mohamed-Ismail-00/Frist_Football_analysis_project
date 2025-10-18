import cv2
import os

# مسار الفيديو
video_path = "C:/Users/HP/Desktop/goal.mp4"


# فولدر لحفظ الإطارات
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# قراءة الفيديو
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # حفظ كل فريم
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"{frame_count}ر'{output_folder}'")

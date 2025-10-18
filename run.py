import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque

# ---------------- إعدادات ----------------
VIDEO_PATH = r"C:\Users\HP\Desktop\foot ball\messi.mp4"
MODEL_PATH = r"C:\Users\HP\Desktop\foot ball\models\best.pt"
OUTPUT_PATH = r"C:\Users\HP\Desktop\foot ball\output_final.mp4"

# الإعدادات البسيطة والفعالة
MOVEMENT_THRESHOLD = 30  # أقل حركة تُحسب (لتجاهل الضوضاء)
PASS_MAX_DISTANCE = 200  # التمريرة: مسافة قصيرة/متوسطة
SHOT_MIN_DISTANCE = 200  # الشوت: مسافة طويلة جدًا

MIN_TIME_BETWEEN_EVENTS = 1.5  # ثانية ونص بين كل حدث
MAX_MISSING_FRAMES = 5  # أقصى إطارات يختفي فيها الكرة ونستمر في التتبع

FPS_FALLBACK = 25.0
MINIMAP_SIZE = (200, 120)

# ---------------- تحميل الموديل ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("❌ فشل في تحميل الفيديو")

fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
font = cv2.FONT_HERSHEY_SIMPLEX

TRAIL_MAX_LEN = int(fps * 3)
MIN_FRAMES_BETWEEN_EVENTS = int(fps * MIN_TIME_BETWEEN_EVENTS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

# ---------------- متغيرات التحليل ----------------
ball_trail = deque(maxlen=TRAIL_MAX_LEN)

pass_count = 0
shot_count = 0

# متغيرات التتبع
tracking_active = False
track_start_pos = None
track_start_frame = 0
total_distance = 0
last_ball_pos = None
missing_frames = 0
last_event_frame = -999

current_frame = 0
current_speed = 0

# ---------------- دوال مساعدة ----------------
def draw_trail(frame, trail):
    for i in range(1, len(trail)):
        pt1 = trail[i - 1]
        pt2 = trail[i]
        if pt1 is None or pt2 is None:
            continue
        alpha = 1 - (i / len(trail))
        color = (0, int(255 * alpha), int(255 * alpha))
        cv2.line(frame, pt1, pt2, color, 3)
    return frame

def draw_minimap(frame, detections, ball_center):
    minimap = np.zeros((MINIMAP_SIZE[1], MINIMAP_SIZE[0], 3), dtype=np.uint8)
    minimap[:] = (0, 100, 0)
    cv2.rectangle(minimap, (0, 0), (MINIMAP_SIZE[0]-1, MINIMAP_SIZE[1]-1), (255, 255, 255), 1)

    for det in detections:
        cls_id = int(det[5])
        x1, y1, x2, y2 = det[:4]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        map_x = int(center_x * MINIMAP_SIZE[0] / w)
        map_y = int(center_y * MINIMAP_SIZE[1] / h)
        
        if cls_id == 0:
            cv2.circle(minimap, (map_x, map_y), 3, (255, 255, 255), -1)
        elif cls_id == 1:
            cv2.circle(minimap, (map_x, map_y), 3, (255, 0, 0), -1)
        elif cls_id == 2:
            cv2.circle(minimap, (map_x, map_y), 3, (0, 0, 255), -1)
        elif cls_id == 3:
            cv2.circle(minimap, (map_x, map_y), 3, (0, 255, 255), -1)

    if ball_center:
        map_x = int(ball_center[0] * MINIMAP_SIZE[0] / w)
        map_y = int(ball_center[1] * MINIMAP_SIZE[1] / h)
        cv2.circle(minimap, (map_x, map_y), 3, (255, 255, 255), -1)

    frame[-MINIMAP_SIZE[1]:, -MINIMAP_SIZE[0]:, :] = minimap
    return frame

def finalize_event(distance, start_frame, end_frame):
    """تحديد نوع الحدث بناءً على المسافة المقطوعة"""
    global pass_count, shot_count, last_event_frame
    
    # تجاهل الحركات الصغيرة
    if distance < MOVEMENT_THRESHOLD:
        return
    
    # تجاهل الأحداث المتقاربة جدًا
    if (end_frame - last_event_frame) < MIN_FRAMES_BETWEEN_EVENTS:
        return
    
    # الحكم البسيط: المسافة الطويلة = شوت، القصيرة = تمريرة
    if distance >= SHOT_MIN_DISTANCE:
        shot_count += 1
        last_event_frame = end_frame
        print(f"🎯 SHOT! Frame: {end_frame} | Distance: {distance:.0f} px | Duration: {(end_frame-start_frame)/fps:.2f}s")
    
    elif distance >= MOVEMENT_THRESHOLD:
        pass_count += 1
        last_event_frame = end_frame
        print(f"⚽ PASS! Frame: {end_frame} | Distance: {distance:.0f} px | Duration: {(end_frame-start_frame)/fps:.2f}s")

# ---------------- الحلقة الرئيسية ----------------
print("\n🎬 بدء التحليل...")
print("="*60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1

    results = model(frame, verbose=False)[0]
    detections = results.boxes.data.cpu().numpy()

    ball_center = None

    for det in detections:
        cls_id = int(det[5])
        x1, y1, x2, y2 = det[:4]

        if cls_id == 0:
            ball_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(frame, ball_center, 8, (0, 255, 255), -1)
        elif cls_id in [1, 2, 3]:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # ---------- المنطق الأساسي ----------
    if ball_center:
        missing_frames = 0
        ball_trail.appendleft(ball_center)
        
        if last_ball_pos is not None:
            frame_distance = math.hypot(
                ball_center[0] - last_ball_pos[0],
                ball_center[1] - last_ball_pos[1]
            )
            current_speed = frame_distance * fps
            
            # بداية تتبع جديد
            if not tracking_active:
                if frame_distance > MOVEMENT_THRESHOLD:
                    tracking_active = True
                    track_start_pos = last_ball_pos
                    track_start_frame = current_frame
                    total_distance = frame_distance
                    print(f"▶️  Tracking started at frame {current_frame}")
            
            # استمرار التتبع
            else:
                total_distance += frame_distance
        
        last_ball_pos = ball_center
        frame = draw_trail(frame, ball_trail)
    
    else:
        # الكرة اختفت
        missing_frames += 1
        
        # إنهاء التتبع لو الاختفاء طال
        if tracking_active and missing_frames > MAX_MISSING_FRAMES:
            finalize_event(total_distance, track_start_frame, current_frame - missing_frames)
            tracking_active = False
            total_distance = 0
            print(f"⏹️  Tracking ended (ball lost)\n")
    
    # إنهاء التتبع لو الكرة بطئت/توقفت
    if tracking_active and ball_center and current_speed < 100:
        finalize_event(total_distance, track_start_frame, current_frame)
        tracking_active = False
        total_distance = 0
        print(f"⏹️  Tracking ended (ball stopped)\n")

    # ---------- رسم الخريطة الصغيرة ----------
    frame = draw_minimap(frame, detections, ball_center)

    # ---------- رسم الداشبورد ----------
    dash_h = 80
    dashboard = np.zeros((dash_h, w-MINIMAP_SIZE[0], 3), dtype=np.uint8)

    section_width = (w - MINIMAP_SIZE[0]) // 3
    
    cv2.putText(dashboard, f"Passes: {pass_count}", (30, 45), font, 1, (0, 255, 0), 2)
    cv2.putText(dashboard, f"Shots: {shot_count}", (section_width + 30, 45), font, 1, (0, 255, 255), 2)
    
    # عرض المسافة الحالية أثناء التتبع
    if tracking_active:
        cv2.putText(dashboard, f"Track: {total_distance:.0f}px", (section_width * 2 + 30, 45), font, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(dashboard, f"Speed: {current_speed:.0f}", (section_width * 2 + 30, 45), font, 0.8, (255, 255, 0), 2)

    frame[-dash_h:, :w-MINIMAP_SIZE[0], :] = cv2.addWeighted(frame[-dash_h:, :w-MINIMAP_SIZE[0], :], 0.3, dashboard, 0.7, 0)

    out.write(frame)

    cv2.imshow("Football Analytics", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# إنهاء أي تتبع نشط
if tracking_active:
    finalize_event(total_distance, track_start_frame, current_frame)

# ---------------- إنهاء البرنامج ----------------
cap.release()
out.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("✅ انتهى التحليل بنجاح!")
print(f"📁 الفيديو محفوظ في: {OUTPUT_PATH}")
print(f"\n📊 النتائج النهائية:")
print(f"   ⚽ Passes: {pass_count}")
print(f"   🎯 Shots: {shot_count}")
print("="*60)
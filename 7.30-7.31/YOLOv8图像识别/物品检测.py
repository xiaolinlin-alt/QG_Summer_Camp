from ultralytics import YOLO
import cv2
import time

# 加载模型（自动下载最新权重）
model = YOLO('yolo12n.pt')

# 摄像头初始化
cap = cv2.VideoCapture(0)  # 0表示默认摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 实时检测循环
fps_start_time = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 计算FPS
    fps_end_time = time.time()
    fps = 1 / (fps_end_time - fps_start_time)
    fps_start_time = fps_end_time

    # 模型推理（设置置信度阈值0.4）
    results = model(frame, conf=0.4, imgsz=640)#太低的阈值不要

    # 结果可视化
    annotated_frame = results[0].plot()
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)#把FPS那段文字写在窗口左上角，绿色

    # 显示画面
    cv2.imshow("YOLOv12 Real-time Detection", annotated_frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()#释放摄像头句柄
cv2.destroyAllWindows()#销毁所有highGUI窗口
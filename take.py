import cv2
import os
import time
from picamera2 import Picamera2

# 1. 設定學號/名稱與資料儲存路徑
name = input("請輸入您的姓名或學號（例如 student01）: ")
save_dir = f"/home/pi/411306171/LBPH_Face/data/{name}"
os.makedirs(save_dir, exist_ok=True)

# 2. 初始化 Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

target_count = 20
print(f"\n準備開始自動拍照（已加入上下翻轉修正），將為 {name} 收集 {target_count} 張照片。")
print("程式將在 5 秒後開始，請準備好面對鏡頭...")

# 倒數 5 秒
for i in range(5, 0, -1):
    print(f"倒數 {i} 秒...")
    time.sleep(1)

print("開始拍照！")

try:
    for count in range(1, target_count + 1):
        # 擷取畫面
        frame_rgb = picam2.capture_array()
        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 修正上下顛倒（參數 0 代表上下翻轉 / Flip Vertically）
        img_bgr = cv2.flip(img_bgr, 0)
        
        # 轉為灰階（LBPH 訓練通常使用灰階影像）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 儲存照片
        file_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(file_path, gray)
        print(f"[{count}/{target_count}] 已成功拍照、翻轉並儲存至: {file_path}")
        
        # 若不是最後一張，則等待 2 秒再拍下一張
        if count < target_count:
            time.sleep(2)

finally:
    picam2.stop()
    print(f"\n拍攝結束！共成功收集 {target_count} 張方向正確的照片。")
import cv2
import os
import urllib.request
from picamera2 import Picamera2

# 1. 自動檢查並下載 Haar 模型（確保辨識時能抓出人臉位置）
haar_filename = 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_filename):
    print("正在下載人臉偵測模型檔...")
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, haar_filename)

face_cascade = cv2.CascadeClassifier(haar_filename)

# 2. 載入訓練好的 LBPH 模型
model_path = "/home/pi/411306171/LBPH_Face/models/lbph_model.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(model_path):
    recognizer.read(model_path)
    print("成功載入 LBPH 人臉辨識模型！")
else:
    print(f"警告：找不到模型檔 {model_path}，請確認是否已完成訓練。")

# 3. 標籤對應表（根據剛剛訓練結果：標籤 0 對應 st1）
label_dict = {
    0: "st1"
}

# 4. 初始化 Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

print("\n開始進行即時人臉辨識，由於是在遠端或無圖形介面環境，程式會自動將辨識畫面儲存或在終端機顯示。若有本機視窗可按 'q' 離開。")

try:
    while True:
        frame_rgb = picam2.capture_array()
        img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 修正畫面方向（如果拍照時有上下翻轉，辨識時也需要對應翻轉）
        img_bgr = cv2.flip(img_bgr, 0)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            
            # 進行預測
            label, confidence = recognizer.predict(face_resized)
            
            # 判斷相似度（數值越低越相似，小於 80 視為認得）
            if confidence < 100:
                name = label_dict.get(label, "Unknown")
                text = f"{name} ({confidence:.1f})"
                color = (0, 255, 0) # 綠色框
            else:
                text = f"Unknown ({confidence:.1f})"
                color = (0, 0, 255) # 紅色框
                
            # 畫出外框與標籤
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            print(f"偵測到人臉 -> 辨識結果: {text}")

        # 如果在樹莓派本機桌面環境可取消下面註解來顯示視窗
        # cv2.imshow("LBPH Face Recognition", img_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\n程式被使用者中斷。")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("人臉辨識程式已安全關閉。")
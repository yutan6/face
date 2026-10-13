import cv2
import os
import numpy as np

# 定義資料與模型存放路徑
data_dir = "/home/pi/411306171/LBPH_Face/data"
models_dir = "/home/pi/411306171/LBPH_Face/models"
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, "lbph_model.yml")

print("正在準備訓練資料...")

faces = []
labels = []
label_dict = {}
current_label = 0

# 檢查 data 資料夾是否存在
if not os.path.exists(data_dir):
    print(f"錯誤：找不到資料夾 {data_dir}，請先確認是否有收集人臉照片。")
else:
    # 走訪 data 資料夾底下的每個人物資料夾
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        
        if not os.path.isdir(person_path):
            continue
            
        # 為每個人建立一個數值編號 (Label)
        label_dict[current_label] = person_name
        print(f"發現使用者: {person_name} -> 對應標籤編號: {current_label}")
        
        # 讀取該資料夾內的所有圖片
        for img_name in os.listdir(person_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(person_path, img_name)
                # 讀取灰階影像
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # 確保圖片尺寸一致 (200x200)
                    img_resized = cv2.resize(img, (200, 200))
                    faces.append(img_resized)
                    labels.append(current_label)
                    
        current_label += 1

    if len(faces) > 0:
        # 初始化 LBPH 人臉辨識模型
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        print("正在進行 LBPH 模型訓練，請稍候...")
        recognizer.train(faces, np.array(labels))
        
        # 儲存訓練好的模型
        recognizer.save(model_path)
        print(f"模型訓練成功！已儲存至: {model_path}")
        
        # 顯示標籤對應關係（辨識程式會需要用到這個對應表）
        print("\n[辨識標籤對應表]")
        for label_id, name in label_dict.items():
            print(f"標籤 {label_id} : {name}")
            
    else:
        print("錯誤：沒有找到任何有效的訓練圖片，請確認照片是否有成功拍入資料夾中。")
import cv2
import numpy as np
import os

# 設定路徑
input_path = "/home/pi/411306171/input.jpg"
output_dir = "/home/pi/411306171"

# 1. 讀取影像
img_bgr = cv2.imread(input_path)

if img_bgr is None:
    print(f"錯誤：無法讀取位於 {input_path} 的圖片，請確認檔案名稱與路徑。")
else:
    # 定義 16 色 / 256 色量化函數（K-Means 演算法）
    def quantize_color(img, k):
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        return quantized.reshape(img.shape)

    print("正在轉換灰階與黑白影像...")
    # 2. 灰階影像
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 3. 黑白影像 (二值化)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    
    # 4. 256 色與 16 色影像處理
    print("正在進行 256 色與 16 色色彩量化計算...")
    img_256 = quantize_color(img_bgr, 256)
    img_16 = quantize_color(img_bgr, 16)

    # 儲存結果到指定資料夾中
    cv2.imwrite(os.path.join(output_dir, "output_gray.jpg"), gray_img)
    cv2.imwrite(os.path.join(output_dir, "output_binary.jpg"), binary_img)
    cv2.imwrite(os.path.join(output_dir, "output_256.jpg"), img_256)
    cv2.imwrite(os.path.join(output_dir, "output_16.jpg"), img_16)

    print(f"處理完成！所有結果圖片已儲存至: {output_dir}")

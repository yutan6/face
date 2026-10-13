import time
import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string

app = Flask(__name__)

# --- 初始化相機 ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()

# ROI 梯形遮罩座標
pts = np.array([[200, 300], [440, 300], [600, 480], [40, 480]], np.int32)
pts = pts.reshape((-1, 1, 2))

frame_width = 640
xc = frame_width // 2
delta_x = 120

def generate_frames():
    last_save_time = time.time()  # 記錄上次拍照的時間
    
    while True:
        # 1. 擷取畫面
        frame = picam2.capture_array()
        
        # 2. 直接使用相機原始畫面
        img_bgr = frame

        # 3. 旋轉 180 度修正倒裝鏡頭
        img_bgr = cv2.flip(img_bgr, -1)
        
        # 4. 建立繪圖用的空白畫布
        lane_canvas = np.zeros_like(img_bgr)
        
        # 5. 影像處理與車道線偵測
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [pts], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        nonzero_points = cv2.findNonZero(masked_edges)
        x0 = None
        status_text = "No Lane Detected"
        
        if nonzero_points is not None:
            left_lane_points = nonzero_points[nonzero_points[:, :, 0] < xc]
            right_lane_points = nonzero_points[nonzero_points[:, :, 0] >= xc]
            
            has_left = left_lane_points.size > 0
            has_right = right_lane_points.size > 0
            
            # 雙線模式
            if has_left and has_right:
                avg_left_x = np.mean(left_lane_points[:, 0]).astype(int)
                avg_right_x = np.mean(right_lane_points[:, 0]).astype(int)
                cv2.line(lane_canvas, (avg_left_x, 300), (avg_left_x, 480), (0, 255, 0), 8)
                cv2.line(lane_canvas, (avg_right_x, 300), (avg_right_x, 480), (0, 255, 0), 8)
                x0 = (avg_left_x + avg_right_x) // 2
                status_text = "Dual Lines Mode"
                
            # 單右線模式
            elif not has_left and has_right:
                avg_right_x = np.mean(right_lane_points[:, 0]).astype(int)
                cv2.line(lane_canvas, (avg_right_x, 300), (avg_right_x, 480), (0, 255, 0), 8)
                x0 = avg_right_x - delta_x
                status_text = "Single Line - Right"
                
            # 單左線模式
            elif has_left and not has_right:
                avg_left_x = np.mean(left_lane_points[:, 0]).astype(int)
                cv2.line(lane_canvas, (avg_left_x, 300), (avg_left_x, 480), (0, 255, 0), 8)
                x0 = avg_left_x + delta_x
                status_text = "Single Line - Left"

            if x0 is not None:
                cte = xc - x0
                cv2.line(lane_canvas, (x0, 300), (x0, 480), (255, 0, 0), 3)
                cv2.circle(lane_canvas, (x0, 390), 6, (255, 0, 0), -1)
                cv2.putText(lane_canvas, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(lane_canvas, f"CTE: {cte} px", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        combined_img = cv2.addWeighted(img_bgr, 0.7, lane_canvas, 0.3, 0)
        cv2.polylines(combined_img, [pts], True, (0, 0, 255), 2)
        
        # --- 每 5 秒自動儲存一張照片 ---
        current_time = time.time()
        if current_time - last_save_time >= 5.0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, combined_img)
            print(f"[ 自動拍照 ] 已儲存照片: {filename}")
            last_save_time = current_time

        # 串流編碼傳輸
        ret, buffer = cv2.imencode('.jpg', combined_img)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head><title>Raspberry Pi Lane Detection Stream</title></head>
            <body style="background: #111; color: #fff; text-align: center; font-family: sans-serif;">
                <h1>Raspberry Pi 即時車道線與 CTE 偵測串流</h1>
                <div style="margin-top: 20px;">
                    <img src="/video_feed" width="640" height="480" style="border: 2px solid #00f2fe; border-radius: 10px;" />
                </div>
            </body>
        </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
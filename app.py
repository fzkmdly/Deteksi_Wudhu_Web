from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
from flask import jsonify

app = Flask(__name__)

model = YOLO('models/best.pt')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/tentang')
def tentang():
    return render_template('tentang.html')

@app.route('/deteksi')
def deteksi_langsung():
    return render_template('deteksi_langsung.html')

def gen_frames():
    cap = cv2.VideoCapture(0)  # kamera laptop
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(frame, imgsz=640, conf=0.3, device='cpu')
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/belajar')
def belajar():
    return render_template('belajar.html')

@app.route('/belajar/wajah')
def belajar_wajah():
    return render_template('detect/wajah.html')

@app.route('/belajar/tangan')
def belajar_tangan():
    return render_template('detect/lengan.html')

@app.route('/belajar/kepala')
def belajar_kepala():
    return render_template('detect/kepala.html')

@app.route('/belajar/kaki')
def belajar_kaki():
    return render_template('detect/kaki.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Function to generate frames for specific movements
@app.route('/status_deteksi/<gerakan>')
def status_deteksi(gerakan):
    # misalnya kamu punya global last_result dari gen_frames_gerakan
    global last_detected_label
    detected = (last_detected_label == gerakan.lower())
    return jsonify({'terdeteksi': detected})

last_detected_label = None

def gen_frames_gerakan(gerakan):
    global last_detected_label
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            results = model.predict(frame, imgsz=640, conf=0.4)
            names = results[0].names
            boxes = results[0].boxes
            detected_label = None
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id].lower()
                if label == gerakan.lower():
                    detected_label = label
                    break
            last_detected_label = detected_label
            annotated = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed/wajah')
def video_feed_wajah():
    return Response(gen_frames_gerakan('wajah'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/tangan')
def video_feed_tangan():
    return Response(gen_frames_gerakan('tangan'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/kepala')
def video_feed_kepala():
    return Response(gen_frames_gerakan('kepala'), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/kaki')
def video_feed_kaki():
    return Response(gen_frames_gerakan('kaki'), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

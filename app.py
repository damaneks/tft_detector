from flask import Flask, render_template, request, redirect, Response, url_for
from werkzeug.utils import secure_filename

from tft_detection import tft_detector, video_detection
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detected')
def show_detected():
    return render_template('detected.html')

@app.route('/detected_vid')
def show_detected_vid():
    return render_template('detected_vid.html')

@app.route('/detect', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        detection = tft_detector(file_path)

        return redirect('/detected')
    else:   
        return None

@app.route('/detect_vid', methods=['GET', 'POST'])
def upload_vid():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #detection = video_detection(file_path)

        return redirect('/detected_vid')   
    else:
        return None

@app.route('/video_feed')
def video_feed():
    return Response(video_detection('./uploads/test_set5.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
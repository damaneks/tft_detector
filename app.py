from flask import Flask, render_template, request, redirect, Response, url_for
from werkzeug.utils import secure_filename

from vid_detect_feed import vid_detecion_feed
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detected')
def show_detected():
    return render_template('detected.html')

@app.route('/detected_vid/<string:file_name>')
def show_detected_vid(file_name):
    return render_template('detected_vid.html', file_name=file_name)

@app.route('/detect', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

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

        return redirect('/detected_vid/' + secure_filename(f.filename))   
    else:
        return None

@app.route('/video_feed/<string:file_name>')
def video_feed(file_name):
    file_name = 'uploads/' + file_name
    return Response(vid_detecion_feed(file_name), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="192.168.1.21", debug=True)
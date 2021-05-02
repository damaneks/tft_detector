from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename

from tft_detection import tft_detector
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detected')
def show_detected():
    return render_template('detected.html')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
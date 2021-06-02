from flask import Flask, render_template, request, redirect, Response, url_for
from werkzeug.utils import secure_filename

from vid_detect_feed import vid_detecion_feed
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_vid', methods=['GET', 'POST'])
def upload_vid():
    if request.method == 'POST':
        f = request.files['file']
        player = request.form['player']
        place_ended = request.form['place_ended']
        date = request.form['date']
        region = request.form['region']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return render_template('detect_vid.html', file_name=secure_filename(f.filename), player=player, place_ended=place_ended, date=date,
region=region)  
    else:
        return None

@app.route('/video_feed')
def video_feed():
    file_name = 'uploads/' + request.args.get('file_name')
    player = request.args.get('player')
    place_ended = request.args.get('place_ended')
    date = request.args.get('date')
    region = request.args.get('region')
    return Response(vid_detecion_feed(file_name, player, place_ended, date, region), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
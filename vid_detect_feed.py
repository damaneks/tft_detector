import darknet_lib as darknet
import cv2
import json

champs_model_files = {
    'config': "model/champs/yolov4_tft_set5.cfg",
    'data': "model/champs/obj.data",
    'weights': "model/champs/yolov4_tft_set5_best.weights"
}

rounds_model_files = {
    'config': "model/rounds/yolov4_cropped.cfg",
    'data': "model/rounds/cropped.data",
    'weights': "model/rounds/yolov4_cropped_best.weights"
}

counter_model_files = {
    'config': 'model/counter/yolov4_tft_counter.cfg',
    'data':'model/counter/counter.data',
    'weights':'model/counter/yolov4_tft_counter_best.weights'
}

detections_dir = 'detections/'
darknet_width = 512
darknet_height = 512


def vid_detecion_feed(video_path, player, place_ended, date, region):
    rounds_model = {} 
    champs_model = {}
    counter_model = {}
   
    champs_model['network'], champs_model['class_names'], champs_model['class_colors'] = darknet.load_network(
        champs_model_files['config'],
        champs_model_files['data'],
        champs_model_files['weights']
    )

    rounds_model['network'], rounds_model['class_names'], rounds_model['class_colors'] = darknet.load_network(
        rounds_model_files['config'],
        rounds_model_files['data'],
        rounds_model_files['weights']
    )

    counter_model['network'], counter_model['class_names'], counter_model['class_colors'] = darknet.load_network(
        counter_model_files['config'],
        counter_model_files['data'],
        counter_model_files['weights']
    )
    
    input_path = str2int(video_path)
    cap = cv2.VideoCapture(input_path)
    champions = generateChampionsDict(champs_model['class_names'])
    last_5_frames = {
        'frames': [],
        'hasChampionCounter': []
    }
    
    while cap.isOpened():
        for x in range(16):
            ret, frame = cap.read()
        if not ret:
            break
        _, frame_to_web = cv2.imencode('.JPEG', frame)
        if len(last_5_frames['frames']) == 5:
            last_5_frames['frames'].pop(0)
            last_5_frames['hasChampionCounter'].pop(0)
        last_5_frames['frames'].append(frame)

        hasChampionCounter = detectCounter(frame, counter_model)
        last_5_frames['hasChampionCounter'].append(hasChampionCounter)
        
        if last_5_frames['hasChampionCounter'] == [1,1,0,0,0]:
            frame_for_champ_detection = last_5_frames['frames'][0]
            round = detectRound(frame, rounds_model)
            detectChamps(frame_for_champ_detection, champs_model, champions, round)

        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_to_web.tostring() + b'\r\n\r\n')
    cap.release()

    data = readJSONfile()
    data['games'].append({
        'id': len(data['games']) + 1,
        'player': player,
        'place_ended': place_ended,
        'date': date,
        'region': region,
        'champions': champions
    })
    saveJSONfile(data)

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def str2int(video_path):
    try:
        return int(video_path)
    except ValueError:
        return video_path

def readJSONfile():
    with open(detections_dir + 'detections.json') as json_file:
        return json.load(json_file)

def saveJSONfile(data):
    with open(detections_dir + 'detections.json', 'w') as outfile:
        json.dump(data, outfile)

def generateChampionsDict(champs_list):
    champions = {}
    for champ in champs_list:
        champions[champ] = {
            'rounds': [],
            'locations': []
        }
    return champions

def detectCounter(frame, counter_model):
    image = prepImageForDetection(frame)
    detection = darknet.detect_image(counter_model['network'], counter_model['class_names'], image)
    for label, confidence, bbox in detection:
        if 'Champions Counter' in str(label):
            return True
    return False

def detectRound(frame, rounds_model):
    cropped_image = frame[0:512, 704:1216]
    img_for_detection = prepImageForDetection(cropped_image)
    detection = darknet.detect_image(rounds_model['network'], rounds_model['class_names'], img_for_detection, 0.8)
    rounds_list = []
    for label, confidence, bbox in detection:
        x_center = bbox[0]
        rounds_list.append((str(label), x_center))
    rounds_list.sort(key=lambda x:x[1])
    rounds_list = [round[0] for round in rounds_list]
    return getRoundID(rounds_list)

def detectChamps(frame, champs_model, champions_list, round):
    image = prepImageForDetection(frame)
    detection = darknet.detect_image(champs_model['network'], champs_model['class_names'], image, thresh=0.25)

    for label, confidence, bbox in detection:
        print(str(label), bbox)
        if 0.35 < (bbox[1] / darknet_height) < 0.65 and 0.25 < (bbox[0] / darknet_width) < 0.75:
            champions_list[str(label)]['rounds'].append(round)
            champions_list[str(label)]['locations'].append(bbox)


def prepImageForDetection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes()) 
    return img_for_detect

def getRoundID(rounds_list):
    stage = 0
    round = 0
    last_round = rounds_list[-1]
    if 'Minions' in last_round:
        stage = 1
    elif 'Krugs' in last_round:
        stage = 2
    elif 'Wolves' in last_round:
        stage = 3
    elif 'Raptor' in last_round:
        stage = 4
    elif 'Drake' in last_round:
        stage = 5
    elif 'Herald' in last_round:
        stage = 6
    
    for i in range(len(rounds_list)):
        if 'active' in rounds_list[i]:
            round = i + 1
    
    return str(stage) + '-' + str(round)
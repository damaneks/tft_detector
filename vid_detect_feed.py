import darknet_lib as darknet
import cv2
import json

champs_model_files = {
    'config': "model/champs/yolov4_tft_set5.cfg",
    'data': "model/champs/obj.data",
    'weights': "model/champs/yolov4_tft_set5_best.weights"
}

rounds_model_files = {
    'config': "model/rounds/yolov4_rounds.cfg",
    'data': "model/rounds/rounds.data",
    'weights': "model/rounds/yolov4_rounds_best.weights"
}

counter_model_files = {
    'config': 'model/counter/yolov4_tft_counter.cfg',
    'data':'model/counter/counter.data',
    'weights':'model/counter/yolov4_tft_counter_best.weights'
}

detections_dir = 'detections/'
thresh=0.25
#darknet_width = 416
#darknet_height = 416
darknet_width = 512
darknet_height = 512


def vid_detecion_feed(video_path, player, place_ended, date, region):
    rounds_model = {} 
    champs_model = {}
    counter_model = {}
   
    champs_model['network'], champs_model['class_names'], champs_model['class_colors'] = darknet.load_network(
        champs_model_files['config'],
        champs_model_files['data'],
        champs_model_files['weights'],
        batch_size=1
    )

    rounds_model['network'], rounds_model['class_names'], rounds_model['class_colors'] = darknet.load_network(
        rounds_model_files['config'],
        rounds_model_files['data'],
        rounds_model_files['weights'],
        batch_size=1
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
    round_number = 0
    while cap.isOpened():
        for x in range(16):
            ret, frame = cap.read()
        if not ret:
            break
        _, frame_to_web = cv2.imencode('.JPEG', frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        if len(last_5_frames['frames']) == 5:
            last_5_frames['frames'].pop(0)
            last_5_frames['hasChampionCounter'].pop(0)
        last_5_frames['frames'].append(img_for_detect)
        detection_rounds = darknet.detect_image(rounds_model['network'], rounds_model['class_names'], img_for_detect, thresh)
        #detection_counter = darknet.detect_image(counter_model['network'], counter_model['class_names'], img_for_detect, thresh)
        detections_adjusted = []
        hasChampionCounter = False
        for label, confidence, bbox in detection_rounds:
            bbox_adjusted = convert2original(frame, bbox)
            #print(str(label), bbox)
            if 'Champions Counter' in str(label):
                hasChampionCounter = True
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        #print('\n')
        last_5_frames['hasChampionCounter'].append(hasChampionCounter)
        print(last_5_frames['hasChampionCounter'],'\n')
        if last_5_frames['hasChampionCounter'] == [1,1,0,0,0]:
            round_number += 1
            img_for_detect_champs = last_5_frames['frames'][0]
            detection_champs = darknet.detect_image(champs_model['network'], champs_model['class_names'], img_for_detect_champs, thresh)
            detections_adjusted_champs = []
            for label_champs, confidence_champs, bbox_champs in detection_champs:
                if 0.35 < (bbox_champs[1]/darknet_height) < 0.65 and 0.25 < (bbox_champs[0]/darknet_width) < 0.75:
                    bbox_adjusted_champs = convert2original(frame, bbox_champs)
                    print(label_champs)
                    champions[str(label_champs)]['rounds'].append(round_number)
                    champions[str(label_champs)]['locations'].append(bbox_champs)
                    detections_adjusted_champs.append((str(label_champs), confidence_champs, bbox_adjusted_champs))
            image = darknet.draw_boxes(detections_adjusted_champs, frame, champs_model['class_colors'])
            #_, frame_to_web = cv2.imencode('.JPEG', image)


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
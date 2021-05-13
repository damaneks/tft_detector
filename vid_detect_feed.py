import darknet_lib as darknet
import cv2
import json

config_file="model/yolov4_tft_set5.cfg"
data_file="model/obj.data"
weights="model/yolov4_tft_set5_best.weights"
detections_dir = 'detections/'
thresh=0.25
darknet_width = 416
darknet_height = 416

def vid_detecion_feed(video_path, player, place_ended, date, region):
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )
    
    input_path = str2int(video_path)
    cap = cv2.VideoCapture(input_path)
    champions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        detection = darknet.detect_image(network, class_names, img_for_detect, thresh)
        
        detections_adjusted = []
        for label, confidence, bbox in detection:
            bbox_adjusted = convert2original(frame, bbox)
            print(str(label), bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        print('\n')
        image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        _, frame_to_web = cv2.imencode('.JPEG', image)
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

def searchChampion(name, champions):
    return [champion for champion in champions if champion['name'] == name]
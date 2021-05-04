import darknet_lib as darknet
import cv2
from flask import make_response

config_file="/home/damian/darknet/cfg/yolov4_tft_set5.cfg"
data_file="/home/damian/darknet/data/obj.data"
weights="/home/damian/darknet/backup/yolov4_tft_set5_best.weights"
output_image_path="uploads/Detection.jpg"
thresh=0.25
darknet_width = 416
darknet_height = 416

def img_detection_feed(image_path):
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size=1
    )

    input_path = str2int(image_path)
    image = cv2.imread(input_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, image_resized.tobytes())
    detection = darknet.detect_image(network, class_names, img_for_detect, thresh)

    detections_adjusted = []
    for label, confidence, bbox in detection:
        bbox_adjusted = convert2original(image, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
    image = darknet.draw_boxes(detections_adjusted, image, class_colors)
    cv2.imwrite(output_image_path, image)
    _, frame_to_web = cv2.imencode('.JPEG', image)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_to_web.tobytes() + b'\r\n\r\n')





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
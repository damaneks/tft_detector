import darknet_lib as darknet
import cv2
import json
import numpy as np
import os

champs_model_files = {
    'config': "model/champs/yolov4_tft_set5.cfg",
    'data': "model/champs/obj.data",
    'weights': "model/champs/yolov4_champs_best.weights"
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
    rounds = {}
    last_10_frames = {
        'frames': [],
        'hasChampionCounter': []
    }
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counter = 0
    detectedRound = '0-0'
    championsInRound = []
    delay = 16
    while cap.isOpened():
        for x in range(delay):
            ret, frame = cap.read()
            frame_counter += 1
        if not ret:
            break
        if len(last_10_frames['frames']) == 10:
            last_10_frames['frames'].pop(0)
            last_10_frames['hasChampionCounter'].pop(0)
        last_10_frames['frames'].append(frame)

        hasChampionCounter = detectCounter(frame, counter_model)
        last_10_frames['hasChampionCounter'].append(hasChampionCounter)
        if hasChampionCounter:
            delay = 16
        else:
            delay = 64
        if last_10_frames['hasChampionCounter'] == [1,1,1,1,1,0,0,0,0,0]:
            frame_for_champ_detection = last_10_frames['frames'][0:5]
            if rounds == {}:
                detectedRound, championsInRound = detectChampsForRound(frame_for_champ_detection, champs_model, rounds_model, '0-0')
            else:
                detectedRound, championsInRound = detectChampsForRound(frame_for_champ_detection, champs_model, rounds_model, max(rounds))
            if detectedRound in rounds:
                if len(rounds[detectedRound]) < len(championsInRound): 
                    rounds[detectedRound] = championsInRound
            else:
                rounds[detectedRound] = championsInRound

        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.JPEG', genTrackBar(detectedRound, championsInRound, frame_counter/number_of_frames))[1].tobytes() + b'\r\n\r\n')


    cap.release()
    os.remove(video_path)
    #locations_file = readJSONfile('locations.json')
    #locations_file = generateLocationsDict(champs_model['class_names'])
    #saveJSONfile(locations_file, 'locations.json')
    #updateLocationsJSON(rounds, locations_file)

    saveJSONfile(generatePopularityDict(champs_model['class_names']), 'popularity.json')

    # for champ in champs_model['class_names']:
    #     champ = champ.lower()
    #     champ = ''.join(champ.split(" "))
    #     champ = ''.join(champ.split("'"))
    #     saveJSONfile(generatePopularityDictForChamp(), 'champions/' + champ + '.json')

    #updatePopularityJSON(rounds, region, player)
    data = readJSONfile('detections.json')
    data['games'].append({
        'id': len(data['games']) + 1,
        'player': player,
        'place_ended': place_ended,
        'date': date,
        'region': region,
        'rounds': rounds
    })
    saveJSONfile(data, 'detections.json')
    print('END')

def genTrackBar(detectedRound, championsInRound, progress):
    trackBar = np.zeros((600,512,3), np.uint8)
    cv2.rectangle(trackBar, (0,0), (512, 40), (0,0,255), -1)
    progress_width = (int(512 * progress), 40)
    cv2.rectangle(trackBar, (0,0), progress_width, (0,255,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(trackBar, 'Last detection:', (10, 100),font, 2, (255,255,255), 5)
    cv2.putText(trackBar, 'Round: ' + detectedRound, (10, 155),font, 2, (255,255,255), 3)
    if len(championsInRound) > 0:
        y = 185
        for line in championsInRound:
            cv2.putText(trackBar, championsInRound[line], (10, y),font, 1, (255,255,255), 1)
            y = y + 30

    return trackBar

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

def readJSONfile(file_name):
    with open(detections_dir + file_name) as json_file:
        return json.load(json_file)

def saveJSONfile(data, file_name):
    with open(detections_dir + file_name, 'w') as outfile:
        json.dump(data, outfile, indent=2)

def generateChampionsDict(champs_list):
    champions = {}
    for champ in champs_list:
        champions[champ] = {
            'rounds': [],
            'locations': []
        }
    return champions

def generateLocationsDict(champs_list):
    locationsDict = {}
    locationsDict['global'] = [{}, {}, {}, {}]
    for row in range(4):
        locationsDict['global'][row]['row'] = row + 1
        for column in range(1, 8):
            locationsDict['global'][row]['column' + str(column)] = 0
    for champ in champs_list:
        champ = champ.lower()
        champ = ''.join(champ.split(" "))
        champ = ''.join(champ.split("'"))
        locationsDict[champ] = [{}, {}, {}, {}]
        for row in range(4):
            locationsDict[champ][row]['row'] = row + 1
            for column in range(1, 8):
                locationsDict[champ][row]['column' + str(column)] = 0
    return locationsDict

def generatePopularityDict(champs_list):
    popularityDict = {}
    for champ in champs_list:
        champ = champ.lower()
        champ = ''.join(champ.split(" "))
        champ = ''.join(champ.split("'"))
        popularityDict[champ] = {}
        popularityDict[champ]['id'] = champ
        popularityDict[champ]['data'] = []
        for stage in range(2, 8):
            for round in range(1, 7):
                if round != 4:
                    popularityDict[champ]['data'].append({
                        'x': str(stage) + '-' + str(round),
                        'y': 0
                    })
    return popularityDict

def generatePopularityDictForChamp():
    popularityDict = {}
    regions = ['Europe', 'Asia', 'America']
    players = ['Zbrojson', 'Laellana','KC_Double61','bebe872', 'Kiyoon', 'pockygom', 'NoobOwl', 'default']
    regionsAndPlayers = regions + players
    for playerOrRegion in regionsAndPlayers:
        popularityDict[playerOrRegion] = {}
        popularityDict[playerOrRegion]['id'] = playerOrRegion
        popularityDict[playerOrRegion]['data'] = []
        for stage in range(2, 8):
            for round in range(1, 7):
                if round != 4:
                    popularityDict[playerOrRegion]['data'].append({
                        'x': str(stage) + '-' + str(round),
                        'y': 0
                    })
    return popularityDict


def updateLocationsJSON(roundsDict, locationsDict):
    for round in roundsDict:
        for locationTile in roundsDict[round]:
            champion = roundsDict[round][locationTile]
            champion = champion.lower()
            champion = ''.join(champion.split(" "))
            champion = ''.join(champion.split("'"))
            row = locationTile.split('-')[0]
            column = locationTile.split('-')[1]
            if locationsDict['global'][int(row) - 1]['column' + str(column)] > 0:
                locationsDict['global'][int(row) - 1]['column' + str(column)] = locationsDict['global'][int(row) - 1]['column' + str(column)] + 1
            else:
                locationsDict['global'][int(row) - 1]['column' + str(column)] = 1
            if locationsDict[champion][int(row) - 1]['column' + str(column)] > 0:
                locationsDict[champion][int(row) - 1]['column' + str(column)] = locationsDict[champion][int(row) - 1]['column' + str(column)] + 1
            else:
                locationsDict[champion][int(row) - 1]['column' + str(column)] = 1
        
    saveJSONfile(locationsDict, 'locations.json')

def updatePopularityJSON(roundsDict, region, player):
    regionDict = readJSONfile('regions/' + region.lower() + '.json')
    playerDict = readJSONfile('players/' + player.lower() + '.json')
    popularityDict = readJSONfile('popularity.json')
    for round in roundsDict:
        for locationTile in roundsDict[round]:
            champion = roundsDict[round][locationTile]
            champion = champion.lower()
            champion = ''.join(champion.split(" "))
            champion = ''.join(champion.split("'"))
            stageNumber = int(round.split('-')[0])
            roundNumber = int(round.split('-')[1])
            roundIndex = 5 * (stageNumber-2) + (roundNumber - 1)
            print(round)
            print(roundIndex)
            regionDict[champion]['data'][roundIndex]['y'] = regionDict[champion]['data'][roundIndex]['y'] + 1
            playerDict[champion]['data'][roundIndex]['y'] = playerDict[champion]['data'][roundIndex]['y'] + 1
            popularityDict[champion]['data'][roundIndex]['y'] = popularityDict[champion]['data'][roundIndex]['y'] + 1
            championDict = readJSONfile('champions/' + champion + '.json')
            championDict[region]['data'][roundIndex]['y'] = championDict[region]['data'][roundIndex]['y'] + 1
            championDict[player]['data'][roundIndex]['y'] = championDict[player]['data'][roundIndex]['y'] + 1
            saveJSONfile(championDict, 'champions/' + champion + '.json')
    saveJSONfile(regionDict, 'regions/' + region.lower() + '.json')
    saveJSONfile(playerDict, 'players/' + player.lower() + '.json')

def detectCounter(frame, counter_model):
    image = prepImageForDetection(frame)
    detection = darknet.detect_image(counter_model['network'], counter_model['class_names'], image)
    for label, confidence, bbox in detection:
        if 'Champions Counter' in str(label):
            return True
    return False

def detectRound(frame, rounds_model, lastRound):
    cropped_image = frame[0:512, 704:1216]
    img_for_detection = prepImageForDetection(cropped_image)
    detection = darknet.detect_image(rounds_model['network'], rounds_model['class_names'], img_for_detection, 0.8)
    rounds_list = []
    for label, confidence, bbox in detection:
        x_center = bbox[0]
        rounds_list.append((str(label), x_center))
    rounds_list.sort(key=lambda x:x[1])
    rounds_list = [round[0] for round in rounds_list]
    return getRoundID(rounds_list, lastRound)

def detectChamps(frame, champs_model):
    image = prepImageForDetection(frame)
    detection = darknet.detect_image(champs_model['network'], champs_model['class_names'], image, thresh=0.65)
    detectedChampsInFrame = []
    for label, confidence, bbox in detection:
        if 0.35 < (bbox[1] / darknet_height) < 0.65 and 0.25 < (bbox[0] / darknet_width) < 0.75:
            location = (bbox[0], bbox[1] + bbox[3]/2)
            array = []
            array.append(label)
            array.append(getLocationID(location))
            array.append(confidence)
            detectedChampsInFrame.append(array)
    print(detectedChampsInFrame)
    return detectedChampsInFrame 

def detectChampsForRound(frames, champs_model, rounds_model, lastRound):
    detectedChampsList = []
    roundsList = []
    for frame in frames:
        detectedChampsList.append(detectChamps(frame, champs_model))
        roundsList.append(detectRound(frame, rounds_model, lastRound))
    
    champCounter = {}
    for detectedChampsInFrame in detectedChampsList:
        for champ in detectedChampsInFrame:
            if champ[0] in champCounter.keys():
                champCounter[champ[0]] = champCounter[champ[0]] + 1
            else:
                champCounter[champ[0]] = 1

    championInRound = {}
    for champName in champCounter:
        if champCounter[champName] >= 3:
            counter = int(champCounter[champName] / 3)
            if counter > 2:
                counter = 2
            for i in range(4, -1, -1):
                for champ in detectedChampsList[i]:
                    if champ[0] == champName:
                        if champ[1] not in championInRound.keys():
                            championInRound[champ[1]] = champ[0]
                            counter = counter - 1
                if counter == 0:
                    break
    
    detectedRound = max(set(roundsList), key=roundsList.count)
    print(detectedRound)
    print(championInRound)
    return detectedRound, championInRound

def prepImageForDetection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes()) 
    return img_for_detect

def getRoundID(rounds_list, last_round_detected):
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

    if 'Box active' in rounds_list:
        round = 2
    elif 'Minions active' in rounds_list:
        round = 2
        for round_name in rounds_list:
            if 'Minions won' in round_name or 'Minions lost' in round_name:
                round = round + 1
    elif 'active' in rounds_list[-1]:
        round = 7
    elif 'PVP active' in rounds_list:
        if stage < 5:
            round = 1
            for round_name in rounds_list:
                if 'PVP won' in round_name or 'PVP lost' in round_name:
                    round = round + 1
            if round >= 2:
                 round = round + 1
            if round >= 4:
                round = round + 1
        else:
            round = 1
            for round_name in rounds_list:
                if 'PVP won' in round_name or 'PVP lost' in round_name:
                    round = round + 1
            if round >= 4:
                round = round + 1
    if '7-' in last_round_detected:
        stage = 7

    if stage == 6 and round == 1 and last_round_detected > '6-1':
        stage = 7

    if last_round_detected > str(stage) + '-' + str(round):
        return last_round_detected

    return str(stage) + '-' + str(round)

def getLocationID(location_list):
    row = 0
    column = 0
    [x, y] = location_list
    if y < 230:
        row = 1
        x = x + 18
    elif y < 275:
        row = 2
    elif y < 320:
        row = 3
        x = x + 18
    else:
        row = 4
    
    if x < 170:
        column = 1
    elif x < 205:
        column = 2
    elif x < 240:
        column = 3
    elif x < 275:
        column = 4
    elif x < 310:
        column = 5
    elif x < 345:
        column = 6
    else:
        column = 7
    
    return str(row) + '-' + str(column)
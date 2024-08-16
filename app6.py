from flask import Flask, render_template, send_file, request, jsonify
import cv2
import torch
from ultralytics import YOLO
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from extra import detect_and_announce_precise, get_distance

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_path = r'/Users/maxyu/Desktop/Max/01_Academic/SE/Project/yolov8n.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路徑未找到: {model_path}")

model = YOLO(model_path).to(device)
print(f"模型加載完成: {model_path}")

class_names = model.names

class_names_zh = {
    'person': '人',
    'bicycle': '自行車',
    'car': '汽車',
    'motorcycle': '摩托車',
    'airplane': '飛機',
    'bus': '公車',
    'train': '火車',
    'truck': '卡車',
    'boat': '船',
    'traffic light': '紅綠燈',
    'fire hydrant': '消防栓',
    'stop sign': '停止標誌',
    'parking meter': '停車收費表',
    'bench': '長椅',
    'bird': '鳥',
    'cat': '貓',
    'dog': '狗',
    'horse': '馬',
    'sheep': '羊',
    'cow': '牛',
    'elephant': '大象',
    'bear': '熊',
    'zebra': '斑馬',
    'giraffe': '長頸鹿',
    'backpack': '背包',
    'umbrella': '雨傘',
    'handbag': '手提包',
    'tie': '領帶',
    'suitcase': '行李箱',
    'frisbee': '飛盤',
    'skis': '滑雪板',
    'snowboard': '單板滑雪',
    'sports ball': '運動球',
    'kite': '風箏',
    'baseball bat': '棒球棒',
    'baseball glove': '棒球手套',
    'skateboard': '滑板',
    'surfboard': '衝浪板',
    'tennis racket': '網球拍',
    'bottle': '瓶子',
    'wine glass': '酒杯',
    'cup': '杯子',
    'fork': '叉子',
    'knife': '刀子',
    'spoon': '湯匙',
    'bowl': '碗',
    'banana': '香蕉',
    'apple': '蘋果',
    'sandwich': '三明治',
    'orange': '橘子',
    'broccoli': '花椰菜',
    'carrot': '胡蘿蔔',
    'hot dog': '熱狗',
    'pizza': '披薩',
    'donut': '甜甜圈',
    'cake': '蛋糕',
    'chair': '椅子',
    'couch': '沙發',
    'potted plant': '盆栽',
    'bed': '床',
    'dining table': '餐桌',
    'toilet': '馬桶',
    'tv': '電視',
    'laptop': '筆記型電腦',
    'mouse': '滑鼠',
    'remote': '遙控器',
    'keyboard': '鍵盤',
    'cell phone': '手機',
    'microwave': '微波爐',
    'oven': '烤箱',
    'toaster': '烤麵包機',
    'sink': '水槽',
    'refrigerator': '冰箱',
    'book': '書',
    'clock': '時鐘',
    'vase': '花瓶',
    'scissors': '剪刀',
    'teddy bear': '泰迪熊',
    'hair drier': '吹風機',
    'toothbrush': '牙刷'
}

units = {
    'person': '位',
    'bicycle': '輛',
    'car': '輛',
    'motorcycle': '輛',
    'airplane': '架',
    'bus': '輛',
    'train': '列',
    'truck': '輛',
    'boat': '艘',
    'traffic light': '盞',
    'fire hydrant': '支',
    'stop sign': '個',
    'parking meter': '個',
    'bench': '條',
    'bird': '隻',
    'cat': '隻',
    'dog': '隻',
    'horse': '匹',
    'sheep': '隻',
    'cow': '頭',
    'elephant': '頭',
    'bear': '隻',
    'zebra': '隻',
    'giraffe': '隻',
    'backpack': '個',
    'umbrella': '把',
    'handbag': '個',
    'tie': '條',
    'suitcase': '個',
    'frisbee': '個',
    'skis': '雙',
    'snowboard': '塊',
    'sports ball': '個',
    'kite': '個',
    'baseball bat': '根',
    'baseball glove': '隻',
    'skateboard': '塊',
    'surfboard': '塊',
    'tennis racket': '把',
    'bottle': '瓶',
    'wine glass': '個',
    'cup': '個',
    'fork': '把',
    'knife': '把',
    'spoon': '把',
    'bowl': '個',
    'banana': '根',
    'apple': '個',
    'sandwich': '個',
    'orange': '個',
    'broccoli': '棵',
    'carrot': '根',
    'hot dog': '個',
    'pizza': '塊',
    'donut': '個',
    'cake': '塊',
    'chair': '把',
    'couch': '張',
    'potted plant': '盆',
    'bed': '張',
    'dining table': '張',
    'toilet': '個',
    'tv': '台',
    'laptop': '台',
    'mouse': '隻',
    'remote': '個',
    'keyboard': '個',
    'cell phone': '支',
    'microwave': '台',
    'oven': '台',
    'toaster': '台',
    'sink': '個',
    'refrigerator': '台',
    'book': '本',
    'clock': '個',
    'vase': '個',
    'scissors': '把',
    'teddy bear': '隻',
    'hair drier': '個',
    'toothbrush': '支'
}

last_target_name = None

def detect_objects(image, target_name=None):
    global last_target_name
    last_target_name = target_name  # 保存最後一次的目標名稱

    try:
        if not image.startswith("data:image/jpeg;base64,"):
            return {"message": "無效的圖像數據"}
        img_data = base64.b64decode(image.split(",")[1])
        img = Image.open(BytesIO(img_data))
    except Exception as e:
        return {"message": f"加載圖像時出錯: {e}"}

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    height, width, _ = frame.shape
    image_area = height * width

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).to(device).float()
    img = img.permute(2, 0, 1).unsqueeze(0) / 255.0

    results = model(img)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return {"message": "沒有檢測到物體"}

    detected_objects = {}

    for detection in results[0].boxes:
        cls_index = int(detection.cls.item())
        cls_name = class_names[cls_index]
        cls_name_zh = class_names_zh.get(cls_name, cls_name)
        conf = detection.conf.item()

        if conf < 0.5:
            continue

        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        position = get_position((x_min + x_max) / 2, width)
        size = (x_max - x_min) * (y_max - y_min)
        distance = get_distance(size, image_area)

        if target_name:
            target_name_zh = class_names_zh.get(target_name, target_name)
            if cls_name_zh != target_name_zh:
                continue

        if cls_name_zh not in detected_objects:
            detected_objects[cls_name_zh] = {"左邊": {}, "中間": {}, "右邊": {}}

        if position not in detected_objects[cls_name_zh]:
            detected_objects[cls_name_zh][position] = {}

        detected_objects[cls_name_zh][position][distance] = detected_objects[cls_name_zh][position].get(distance, 0) + 1

    if detected_objects:
        detected_text = []
        mp3_list = []
        for obj, positions in detected_objects.items():
            unit = units.get(obj, '個')
            for position, distances in positions.items():
                for distance, count in distances.items():
                    if count > 0:
                        detected_text.append(f"{count} {unit} {obj} 在你的{position}，{distance}")
                        mp3_list.extend([f"count/{count}_{position}.mp3", f"class/{obj}_{position}.mp3", f"pos_dis/{position}{distance}.mp3"])
                        

        detected_text_str = "，".join(detected_text)
        return {"message": detected_text_str, "detections": detected_objects, "mp3_list": mp3_list}
    else:
        return {"message": "未檢測到目標物體"}

def get_position(x_center, width):
    if x_center < width / 3:
        return "左前"
    elif x_center > 2 * width / 3:
        return "右前"
    else:
        return "前方"

@app.route('/')
def index():
    return render_template('t6.html')

@app.route('/play/<folder>/<mp3file>')
def play(folder, mp3file):
    print(f"Playing: {folder}/{mp3file}")
    mp3_path = f"/Users/maxyu/Desktop/Max/01_Academic/SE/Project/main/static/{folder}/{mp3file}"
    if mp3_path and os.path.exists(mp3_path):
        return send_file(mp3_path, mimetype='audio/mpeg')
    else:
        print("mp3檔未找到")
        return {"message": "mp3檔未找到"}

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    image = data['image']
    target_name = data.get('target_name', None)
    precise = data.get('precise', False)

    if precise:
        result = detect_and_announce_precise(image, model, device, class_names, class_names_zh, units, last_target_name)
    else:
        result = detect_objects(image, target_name)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
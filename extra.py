import torch
from PIL import Image
import base64
import numpy as np
from io import BytesIO
import cv2

def get_distance(size, image_area):
    ratio = size / image_area
    return "非常近" if ratio > 1/10 else ("很近" if ratio > 1/20 else "很遠")

def detect_and_announce_precise(image, model, device, class_names, class_names_zh, units, target_name):
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

    # 圖片轉成模型所需格式
    img = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    results = model(img)

    if not results or not results[0].boxes:
        return {"message": "沒有檢測到物體"}

    detected_objects = {}
    for detection in results[0].boxes:
        cls_name = class_names[int(detection.cls.item())]
        cls_name_zh = class_names_zh.get(cls_name, cls_name)
        conf = detection.conf.item()
        if conf < 0.5: continue

        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        x_center = (x_min + x_max) / 2
        if cls_name_zh != target_name or x_center < width / 3 or x_center > 2 * width / 3:
            continue  # 只保留正前方目標物體

        size = (x_max - x_min) * (y_max - y_min)
        distance = get_distance(size, image_area)

        if cls_name_zh not in detected_objects:
            detected_objects[cls_name_zh] = {}
        detected_objects[cls_name_zh][distance] = detected_objects[cls_name_zh].get(distance, 0) + 1

    if detected_objects:
        detected_text = []
        for obj, distances in detected_objects.items():
            for distance, count in distances.items():
                if count > 0:
                    detected_text.append(f"你前方有{count}{units.get(obj, '個')}{obj}，{distance}")

        return {"message": "，".join(detected_text), "detections": detected_objects}
    return {"message": "未檢測到目標物體."}
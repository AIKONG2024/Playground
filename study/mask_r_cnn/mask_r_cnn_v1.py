import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
import random
# COCO 클래스
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Mask R-CNN 모델 불러오기
model = maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

def predict(image_path):
    img = Image.open(image_path)
    img_tensor = F.to_tensor(img)
    with torch.no_grad():
        predictions = model([img_tensor])
    return predictions


def visualize(image_path, predictions):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    for score, label, mask, box in zip(predictions[0]['scores'], predictions[0]['labels'], predictions[0]['masks'],
                                       predictions[0]['boxes']):
        if score > 0.5:
            # 객체 마다 다른 색으로 표시
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            # 바운딩 박스 그리기
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=10)

            # 마스크 그리기
            mask = mask[0, :, :].numpy()
            mask = np.array(mask * 100, dtype=np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                draw.polygon([(point[0][0], point[0][1]) for point in contour.tolist()], outline=color, fill=None, width=10)

            # 클래스 이름 표시
            draw.text((box[0], box[1]), COCO_INSTANCE_CATEGORY_NAMES[label], fill='red')

    img.show()


# 이미지 경로 설정
image_path = './horse_person.jpg'

# 예측 수행
predictions = predict(image_path)

# 시각화
visualize(image_path, predictions)
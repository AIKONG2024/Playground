import cv2
import os
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def shift_image_position(img, x_shift, y_shift):
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    shifted_img = cv2.warpAffine(img, M, (cols, rows))
    return shifted_img

def save_images(images, folder):
    for i, img in enumerate(images):
        new_filename = os.path.join(folder, f"shifted_{i}.jpg")
        cv2.imwrite(new_filename, img)

# 이미지를 로드할 디렉토리 경로
input_folder = "C:\Playground\experiment\opencv\\test_image"
# 이미지를 저장할 디렉토리 경로
output_folder = "C:\Playground\experiment\opencv\\arranged_image"

# 이미지 로드
images = load_images_from_folder(input_folder)

# 이미지 재배치
shifted_images = [shift_image_position(img, 100, 50) for img in images]  # 오른쪽으로 100, 아래로 50 픽셀 이동

# 변경된 이미지 저장
save_images(shifted_images, output_folder)
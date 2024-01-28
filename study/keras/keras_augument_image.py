from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#이미지 읽어오기
data_gen = ImageDataGenerator(
    rescale=1./255,
)
image_path = 'C:/_data/image/test_cat/'
data_iter = data_gen.flow_from_directory(
    directory = image_path,
    target_size=(300,300),
    class_mode='binary',
    shuffle=True
)#Found 21 images belonging to 2 classes.

x = data_iter.next()[0] #(x,y) 튜플의 x
y = data_iter.next()[1] #(x,y) 튜플의 y
print(x.shape)#(21, 300, 300, 3)
print(y.shape)#(21, 2)
print(y)
import numpy as np
print(np.unique(y))#[0. 1.]
fig, ax = plt.subplots(nrows=2, ncols=8, figsize = (8,2))
for i in range(16):
    ax[i//8, i%8].imshow(x[i])
    ax[i//8, i%8].axis('off')
plt.show()
#==============================================================================

#이미지 가져오기
from keras.utils import load_img #이미지를 가져옴
from keras.utils import img_to_array #이미지를 수치화
path = "c:/_data/image/test_cat/\Cat/1.jpg"
img = load_img(path, 
            target_size = (300, 300)
)
img_arr = img_to_array(img)
images = np.tile(img_arr.reshape(300*300*3), 6) #이미지 복사
images = images.reshape(-1,300,300,3)


#증강이미지 변형 설정
augument_data_gen = ImageDataGenerator(
    horizontal_flip=True, #좌우반전
    vertical_flip=True, #상하반전
    brightness_range= [0.1, 0.9], #밝기조절
    fill_mode='reflect', #회전,이동,축소시 생긴 공간을 채우는 방식
    zoom_range=0.3, #확대
    rotation_range=30, #회전
    rescale=1./255
)

augument_iter = augument_data_gen.flow(
    images, 
    np.zeros(6), #y값
    batch_size=6, #1배치 자를 사이즈
    shuffle=True #섞기
)

augumented_x = augument_iter.next()[0]
augumented_y = augument_iter.next()[1]

print(augumented_x.shape)
print(augumented_y.shape)

#데이터 합치기
x = np.concatenate((x,augumented_x))
y = np.concatenate((y,augumented_y))


print(x.shape, y.shape) #(27, 300, 300, 3) (27,)
unique , count = np.unique(y, return_counts=True)
print(unique, count) #[0. 1.] [11 11]
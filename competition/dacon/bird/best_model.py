import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D, GlobalAvgPool2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time

SEED = 42
IMAGE_SIZE = (64,64)
FILTERS = 8
BATCH_SIZE = 32
np.random.seed(SEED)
tf.random.set_seed(SEED)

data_path = "./" 
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test.csv")


lbe = LabelEncoder()
lbe.fit_transform(train_df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.3, stratify=train_df['label'], random_state=SEED)

# Data augmentation and data loaders setup
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    # horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='img_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='img_path',
    y_col='label',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,  # 이미지 경로가 절대 경로인 경우 None으로 설정
    x_col='img_path',  # 이미지 경로 열 이름
    y_col=None,  # 테스트 데이터에는 레이블이 없습니다
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,  # 레이블이 없으므로 None으로 설정
    shuffle=False  # 테스트 데이터 순서를 유지해야 합니다
)

num_classes = len(train_generator.class_indices)

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=2, activation='swish', input_shape=(64, 64, 3)))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=48, kernel_size=2))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=24, kernel_size=2, activation='swish'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    model.add(Dense(units=128, activation='swish'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model
    
model = build_model(num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='models/best_model_{epoch}.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
model.summary()
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=3000,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping, model_checkpoint]
)

predictions = model.predict(test_generator, steps=len(test_generator))
predictions = np.argmax(predictions, axis=1)

submission_df = pd.read_csv(data_path + "sample_submission.csv")
submission_df['label'] = lbe.inverse_transform(predictions)
timestr = time.strftime("%Y%m%d-%H%M%S")
submission_df.to_csv(f'sample_submission_{timestr}.csv', index=False)

'''
Value             |Best Value So Far |Hyperparameter
48                |32                |filters_1
5                 |3                 |kernel_size_1
relu              |relu              |activation_1
3                 |1                 |num_layers
232               |80                |units
relu              |tanh              |dense_activation
0.5               |0.4               |dropout
128               |128               |filters_2
7                 |5                 |kernel_size_2
relu              |sigmoid           |activation_2
80                |48                |filters_3
3                 |3                 |kernel_size_3
swish             |relu              |activation_3
0.3               |0.4               |dense_dropout
0.2               |0.3               |dropout_2
0.4               |0.3               |dropout_3
48                |80                |filters_4
5                 |3                 |kernel_size_4
relu              |swish             |activation_4
0.4               |0.2               |dropout_4
2                 |2                 |tuner/epochs
0                 |0                 |tuner/initial_epoch
4                 |4                 |tuner/bracket
0                 |0                 |tuner/round
'''
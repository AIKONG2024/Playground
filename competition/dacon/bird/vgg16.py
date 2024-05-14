import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import math

# Set random seed for reproducibility
SEED = 42
IMAGE_SIZE = (71,71)
FILTERS = 8
BATCH_SIZE = 32
np.random.seed(SEED)
tf.random.set_seed(SEED)

data_path = "./" 
train_df = pd.read_csv(data_path + "train.csv")
test_df = pd.read_csv(data_path + "test.csv")

lbe = LabelEncoder()
lbe.fit_transform(train_df['label'])
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'], random_state=SEED)

# Data augmentation and data loaders setup
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
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

from keras import backend as K
def f1(y_true, y_pred):
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    # x = GlobalAvgPool2D(x)
    x = Flatten(name="flatten")(x)
    x = Dense(64,activation="swish")(x)
    # x = Dropout(0.25)(x)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

num_classes = num_classes = len(train_generator.class_indices)
model = build_model(num_classes)
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', f1])
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

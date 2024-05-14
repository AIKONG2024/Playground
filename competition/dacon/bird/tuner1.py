import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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

upscale_train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='upscale_img_path',
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

upscale_val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='upscale_img_path',
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

class CNNHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential()

        num_layers = hp.Int('num_layers', 1, 3)
        for i in range(1, num_layers):
            model.add(Conv2D(
                filters=hp.Int(f'filters_{i+1}', min_value=16, max_value=128, step=16),
                kernel_size=hp.Choice(f'kernel_size_{i+1}', values=[2]),
                activation=hp.Choice(f'activation_{i+1}', values=['relu', 'tanh', 'sigmoid', 'swish'])
            ))
            model.add(Dropout(rate=hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(
            units=hp.Int('units', min_value=16, max_value=256, step=16),
            activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid', 'swish'])
        ))
        model.add(Dropout(rate=hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

num_classes = len(train_generator.class_indices)
hypermodel = CNNHyperModel(num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='models/best_model_{epoch}.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=100,
    factor=3,
    directory='my_dir',
    project_name='keras_tuner'
)

tuner.search(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint]
)
all_trials = tuner.oracle.trials
max_score = 0.0
best_trial_id = None

for trial_id, trial in all_trials.items():
    if trial.score is not None and max_score < trial.score:
        max_score = trial.score
        best_trial_id = trial_id

# 최고 점수를 가진 트라이얼의 정보만 출력
if best_trial_id is not None:
    best_trial = all_trials[best_trial_id]
    print(f'Best Trial ID: {best_trial_id}, Hyperparameters: {best_trial.hyperparameters.values}, Score: {best_trial.score}')
else:
    print("No trials have completed successfully.")
    
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
history = best_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping, model_checkpoint]
)

predictions = best_model.predict(test_generator, steps=len(test_generator))
predictions = np.argmax(predictions, axis=1)

submission_df = pd.read_csv(data_path + "sample_submission.csv")
submission_df['label'] = lbe.inverse_transform(predictions)
timestr = time.strftime("%Y%m%d-%H%M%S")
submission_df.to_csv(f'sample_submission_{timestr}.csv', index=False)
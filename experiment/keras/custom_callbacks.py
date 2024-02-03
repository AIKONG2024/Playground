# https://www.tensorflow.org/guide/keras/custom_callback?hl=ko
import keras
import numpy as np


# EarlyStopping 뜯어고치기
# monitor: 모니터링 metrics, stop_line: 내려가다가 멈출 기준
class CustomEarlyStoppingAtLoss(keras.callbacks.Callback):
    def __init__(self, patience=0, monitor="loss", stop_line=0.0, is_log=False):
        super(CustomEarlyStoppingAtLoss, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.monitor = monitor
        self.stop_line = stop_line
        self.is_log = is_log

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)        
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
            if self.is_log:
                print(
                f"""
                ❗❗❗early stop renewal❗❗❗
                {self.monitor} : {self.best}
                """
                )

        else:
            if current > self.stop_line :
                # + 추가 )patience 값 도달 전에 monitoring value 가 stop_line 보다 커지면 중단
                self.wait = self.patience
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

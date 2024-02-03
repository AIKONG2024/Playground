# https://www.tensorflow.org/guide/keras/custom_callback?hl=ko
#v1. 24.02.03 KONG SEONUI
import keras
import numpy as np


# EarlyStopping 뜯어고치기

# monitor: 모니터링 metrics
# overfitting_stop_line : 과적합 되면 멈출 기준
# overfitting_count : 과적합 카운팅 기준
class CustomEarlyStoppingAtLoss(keras.callbacks.Callback):
    def __init__(self, patience=0, monitor="loss", overfitting_stop_line=0.0, overfitting_count = 0, is_log=False):
        super(CustomEarlyStoppingAtLoss, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.monitor = monitor
        self.overfitting_stop_line = overfitting_stop_line
        self.is_log = is_log
        self.overfitting_count = overfitting_count
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
                🎊🎊🎊{self.monitor} renewaled🎊🎊🎊
                {self.monitor} : {self.best}
                """
                )

        else:
            # overfittiong_escaping 
            bounce_count = 0
            if current > self.overfitting_stop_line :
                if self.overfitting_count > bounce_count:
                    self.wait = self.patience
                bounce_count += 1      
                          
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

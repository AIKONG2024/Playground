#v1 2024.01.14 KONG SEONUI

from keras_tuner import HyperModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class CustomHyperModel(HyperModel):
    def __init__(self, loss, hidden_activations, last_activation ,metrics):
        self.loss = loss #손실함수
        self.hidden_activations = hidden_activations
        self.last_activation = last_activation #엑티베이션함수
        self.metrics = metrics #
        
    #private    
    def build(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 2,10)): #layer를 0~10
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value = 8,
                                        max_value = 128,
                                        step = 16),
                                        activation=hp.Choice("activation", self.hidden_activations)))
        if self.last_activation:
            model.add(Dense(1, activation=self.last_activation))
        else:
            model.add(Dense(1))
        model.compile(loss = self.loss, optimizer= Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), metrics=self.metrics)
        return model

#선형회귀
class LeanerRegressionModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build(self, hp):
        return CustomHyperModel(loss='mse', hidden_activations = ['relu'], last_activation = None, metrics=['mae']).build(hp)

#이진분류
class BinaryClassificationModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build(self, hp):
        return CustomHyperModel(loss='binary_crossentropy', hidden_activations = ['relu'], last_activation='sigmoid', metrics=['acc']).build(hp)

#다중분류
class MulticlassClassificationModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def build(self, hp):
        return CustomHyperModel(loss='', activation='relu', last_activation='softmax', metrics=['acc']).build(hp)    
    


    


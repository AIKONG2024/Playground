#v1 2024.01.14 KONG SEONUI

from keras_tuner import HyperModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class CustomHyperModel(HyperModel):
    def __init__(self, loss, hidden_activations, last_activation, metrics, output_count):
        self.loss = loss
        self.hidden_activations = hidden_activations
        self.last_activation = last_activation
        self.metrics = metrics
        self.output_count = output_count
        
    #private    
    def build(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 10,25)): #layer를 0~5
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value = 2,
                                        max_value = 50,
                                        step = 2),
                                        activation=hp.Choice("activation", self.hidden_activations)))
        if self.last_activation:
            model.add(Dense(self.output_count, activation=self.last_activation))
        else:
            model.add(Dense(self.output_count))
        model.compile(loss = self.loss, optimizer= Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), metrics=self.metrics)
        return model

#선형회귀
class LeanerRegressionModel(HyperModel):
    def __init__(self, num_classes, output_count):
        self.num_classes = num_classes
        self.output_count = output_count
        
    def build(self, hp):
        return CustomHyperModel(loss='mse', hidden_activations = ['relu'], last_activation = None, metrics=['mae'], output_count= self.output_count).build(hp)

#이진분류
class BinaryClassificationMoadel(HyperModel):
    def __init__(self, num_classes, output_count):
        self.num_classes = num_classes
        self.output_count = output_count
        
    def build(self, hp):
        return CustomHyperModel(loss='binary_crossentropy', hidden_activations = ['relu'], last_activation='sigmoid', metrics=['acc'], output_count= self.output_count).build(hp)

#다중분류
class MulticlassClassificationModel(HyperModel):
    def __init__(self, num_classes, output_count):
        self.num_classes = num_classes
        self.output_count = output_count
        
    def build(self, hp):
        return CustomHyperModel(loss='sparse_categorical_crossentropy',hidden_activations = ['swish'], last_activation='softmax', metrics=['acc'], output_count= self.output_count).build(hp)    
    


    



from keras_tuner import HyperModel
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class LeanerRegressionModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        
    def bulid_model(self, hp):
        model = Sequential()
        for i in range(hp.Int('num_layers', 1,10)): #layer를 1~10
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value = 8,
                                        max_value = 128,
                                        step = 16),
                                        activation=hp.Choice("activation", ["relu"])))
        model.add(Dense(1))
        model.compile(loss = 'mae', optimizer= Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), metrics=['mae'])
        return model
    
    
    
hypermodel = HyperModel(num_classes = 10)
    
    

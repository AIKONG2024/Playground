import numpy as np
from keras.layers import Conv2D, BatchNormalization, Layer, concatenate
from keras.models import Sequential
from keras.activations import swish
from keras.initializers import Constant
from keras.activations import swish

class Conv(Layer):
    def __init__(self,c1,c2,k,s=1, g=1, bias =True, w=None):
        super().__init__()
        Conv2D(
        filters=c2,
        kernel_size=k,
        strides=s,
        padding="VALID",
        use_bias=bias,
        kernel_initializer=Constant(w.weight.permute(2, 3, 1, 0).numpy()),
        bias_initializer=Constant(w.bias.numpy()) if bias else None,
    )
    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs))) #객체 변수를 실행할 때 호출되는 함수
        
class Conv2d(Layer):
    def __init__(self, c2, k, s=1, g=1, bias=True, w=None):
        super().__init__()
        self.conv = Conv2D(
            filters=c2,
            kernel_size=k,
            strides=s,
            padding="VALID",
            use_bias=bias,
            kernel_initializer=Constant(w.weight.permute(2, 3, 1, 0).numpy()),
            bias_initializer=Constant(w.bias.numpy()) if bias else None,
        )
        
    def call(self, inputs):
        return self.conv(inputs)
        
class Bottlenect(Layer):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2
        
    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))

class BN(Layer):
    def __init__(self, w=None):
        super().__init__()
        self.bn = BatchNormalization(
            beta_initializer=Constant(w.bias.numpy()),
            gamma_initializer=Constant(w.weight.numpy()),
            moving_mean_initializer=Constant(w.running_mean.numpy()),
            moving_variance_initializer=Constant(w.running_var.numpy()),
            epsilon=w.eps,)
        
    def call(self, inputs):
        return self.bn(inputs)

class bottlenectCSP(Layer): #Layer 상속
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, w=None): 
        # c1: 입력 dim, ch2:출력 dim , n:해당 모델 번호, 
        # shortcut: 잔차연결여부(기울기 소실 완화/CSP 핵심), 
        #  expansion: 중간층 채널 확장/축소 비율 (0.5면 1/2 축소, 2면 2배 확장)
        super().__init__()
        c_ = int(c2 * e) #중간층 채널 축소(1/2)
        # self.cv1 = Conv(c1, c_, 1,1,w=w.cv1)
        model = Sequential()
        model.add(Conv2D(filters=c1,kernel_size=c_,strides=1,padding="VALID",use_bias=True, activation='swish'))
        model.add(Conv2D(filters=c2,kernel_size=c_,strides=1,padding="VALID",use_bias=True))
        model.add(Conv2D(filters=c2,kernel_size=c_,strides=1,padding="VALID",use_bias=True))
        model.add(Conv2D(filters=c2,kernel_size=c_,strides=1,padding="VALID",use_bias=True))
        self.m = Sequential([Bottlenect(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(concatenate((y1, y2), axis=3))))

c1 = 64
c2 = 128
bottlenect = bottlenectCSP(c1)
bottlenect.summary()
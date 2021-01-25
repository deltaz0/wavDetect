import keras
from keras.layers import Input, Conv1D, TimeDistributed, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np


seqL = 200
fTotalTrain = np.random.rand(1000,32)
filterSize = 32

inputs = Input((seqL,fTotalTrain.shape[1],))
x = Conv1D(filterSize,4,dilation_rate=1,padding='causal',activation='tanh')(inputs)
x = Conv1D(filterSize,4,dilation_rate=2,padding='causal',activation='tanh')(x)
x = Conv1D(filterSize,4,dilation_rate=4,padding='causal',activation='tanh')(x)
x = Conv1D(filterSize,4,dilation_rate=8,padding='causal',activation='tanh')(x)
x = TimeDistributed(Dense(fTotalTrain.shape[1],activation='tanh'))(x)
model = Model(inputs=inputs,outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam')
        
plot_model(model, to_file='model.png', show_shapes=True)
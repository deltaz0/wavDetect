
# coding: utf-8

# In[1]:


import numpy as np
import cPickle as pickle
valInd = 0.2 # (0.0 to 0.8)
pklFileNorm = open('featuresNormal.pkl', 'rb')
pklFileRook = open('featuresRook.pkl', 'rb')
fNorm = pickle.load(pklFileNorm)
fRook = pickle.load(pklFileRook)
pklFileNorm.close()
pklFileRook.close()
valInd = int(fNorm.shape[0] * valInd)
valLen = int(fNorm.shape[0] * 0.2)
valMask = np.zeros((fNorm.shape[0]),bool)
valMask[valInd:valInd+valLen] = True
fNormTrain = fNorm[np.invert(valMask)]
fNormVal = fNorm[valMask]
fRookTrain = fRook[np.invert(valMask)]
fRookVal = fRook[valMask]
del valMask, fNorm, fRook
print 'norm features train shape: ' + str(fNormTrain.shape)
print 'norm features val shape: ' + str(fNormVal.shape)
print 'rook features train shape: ' + str(fRookTrain.shape)
print 'rook features val shape: ' + str(fRookVal.shape)
fRookTrain = np.concatenate((fRookTrain,fNormTrain),axis=0)
print '- Feature Import Cell Complete -'


# In[2]:


from keras.models import Model, load_model
from keras.layers import Dense, Input, LSTM, Conv1D, TimeDistributed
seqL = 200
inputs = Input((seqL,fRookTrain.shape[1],))
x = Conv1D(400,8,dilation_rate=1,padding='causal',activation='tanh')(inputs)
x = TimeDistributed(Dense(fRookTrain.shape[1],activation='tanh'))(x)
model = Model(inputs=inputs,outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam')
print '- Model Init Cell Complete -'


# In[3]:


for i in range(200):
    rI = np.random.randint(fRookTrain.shape[0])
    xTrain = np.roll(fRookTrain,-rI,axis=0)[:seqL*(fRookTrain.shape[0]//seqL)].reshape(-1,seqL,fRookTrain.shape[1])
    yTrain = np.roll(fRookTrain,-(rI+1),axis=0)[:seqL*(fRookTrain.shape[0]//seqL)].reshape(-1,seqL,fRookTrain.shape[1])
    model.fit(xTrain,yTrain,batch_size=xTrain.shape[0]//8,epochs=8)
model.save('modelRookTrain.h5')
del model
print '- Training Cell Complete -'


# In[5]:


model = load_model('modelRookTrain.h5')
l2NormList = []
for i in range(fNormVal.shape[0]//(seqL//16)):
    rI = np.random.randint(fNormVal.shape[0]-(seqL+1))
    pred = model.predict(fNormVal[rI:rI+seqL,:][np.newaxis,:,:])[0]
    l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNormVal[rI+seqL+2-(seqL//2):rI+seqL+2])**2,axis=1)))
l2NormRMS = np.mean(np.array(l2NormList))
print 'norm RMS: ' + str(l2NormRMS)
l2NormList = []
for i in range(fRookVal.shape[0]//(seqL//16)):
    rI = np.random.randint(fRookVal.shape[0]-(seqL+1))
    pred = model.predict(fRookVal[rI:rI+seqL,:][np.newaxis,:,:])[0]
    l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRookVal[rI+seqL+2-(seqL//2):rI+seqL+2])**2,axis=1)))
l2NormRMS = np.mean(np.array(l2NormList))
print 'rook RMS: ' + str(l2NormRMS)
print '- Validation Cell Complete -'


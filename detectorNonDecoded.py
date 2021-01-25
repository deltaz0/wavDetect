
# coding: utf-8

# In[1]:


import numpy as np
import cPickle as pickle
valInd = 0 # (0 to 80)
seqL = 200
pklFileNorm = open('../ncoderdata/featuresNormalNonDecodedExtract.pkl', 'rb')
pklFileRook = open('../ncoderdata/featuresRookNonDecodedExtract.pkl', 'rb')
fNorm = pickle.load(pklFileNorm)
fRook = pickle.load(pklFileRook)
pklFileNorm.close()
pklFileRook.close()
valInd = int(fNorm.shape[0] * (float(valInd)/100.0))
valLen = int(fNorm.shape[0] * 0.25)
valMask = np.zeros((fNorm.shape[0]),bool)
valMask[valInd:valInd+valLen] = True
fNormTrain = fNorm[np.invert(valMask)]
fNormVal = fNorm[valMask]
fRookTrain = fRook[np.invert(valMask)]
fRookVal = fRook[valMask]
fTotalTrain = np.concatenate((fNormVal,fRookVal),axis=0)
del valMask, fNorm, fRook, fNormTrain, fRookTrain, fNormVal, fRookVal
print 'total features train shape: ' + str(fTotalTrain.shape)
#print 'norm features val shape: ' + str(fNormVal.shape)
#print 'rook features val shape: ' + str(fRookVal.shape)
print '- Feature Import Cell Complete -'


# In[ ]:


from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv1D, TimeDistributed
inputs = Input((seqL,fTotalTrain.shape[1],))
x = Conv1D(192,4,dilation_rate=1,padding='causal',activation='tanh')(inputs)
x = Conv1D(192,4,dilation_rate=2,padding='causal',activation='tanh')(x)
x = Conv1D(192,4,dilation_rate=4,padding='causal',activation='tanh')(x)
x = Conv1D(192,4,dilation_rate=8,padding='causal',activation='tanh')(x)
x = TimeDistributed(Dense(fTotalTrain.shape[1],activation='tanh'))(x)
model = Model(inputs=inputs,outputs=x)
model.compile(loss='mean_squared_error', optimizer='adam')
print '- Model Init Cell Complete -'


# In[ ]:


for j in range(20):
    rJ = np.random.randint(fTotalTrain.shape[0])
    for i in range(10):
        rI = rJ * 
        xTrain = np.roll(fTotalTrain,-rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
        yTrain = np.roll(fTotalTrain,-(rI+1),axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
        batchSize = np.random.randint(1,4)
        model.fit(xTrain,yTrain,batch_size=batchSize,epochs=15)
model.save('modelTotalNonDecodedTrain' + str(int(valInd)) + '.h5')
del model
print '- Training Cell Complete -'


# In[ ]:


'''
import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd*10)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
for j in xrange(0,250,1):
    selection = j * 0.003
    l2NormList = []
    fNT = fNormVal[int(selection*(fNormVal.shape[0])):int((selection+0.25)*(fNormVal.shape[0])),:]
    for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
    print 'norm RMS: ' + str(l2NormRMSNorm[j])
    l2NormList = []
    fRT = fRookVal[int(selection*(fRookVal.shape[0])):int((selection+0.25)*(fRookVal.shape[0])),:]
    for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSRook.append(np.mean(np.array(l2NormList)))
    print 'rook RMS: ' + str(l2NormRMSRook[j])
comparisons = np.array(l2NormRMSNorm) - np.array(l2NormRMSRook)
print 'Norm Mean: ' + str(np.mean(np.array(l2NormRMSNorm)))
print 'Rook Mean: ' + str(np.mean(np.array(l2NormRMSRook)))
plt.plot(comparisons)
plt.show()
plt.clf()
print '- Validation Cell Complete -'
'''


# In[ ]:


'''
import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd*10)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
rList = np.arange(500)
np.random.shuffle(rList)
for j in xrange(0,250,1):
    selection = rList[j] * 0.0015
    l2NormList = []
    fNT = fNormVal[int(selection*(fNormVal.shape[0])):int((selection+0.25)*(fNormVal.shape[0])),:]
    for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
    print 'norm RMS: ' + str(l2NormRMSNorm[j])
    selection = rList[j+250] * 0.0015
    l2NormList = []
    fRT = fRookVal[int(selection*(fRookVal.shape[0])):int((selection+0.25)*(fRookVal.shape[0])),:]
    for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSRook.append(np.mean(np.array(l2NormList)))
    print 'rook RMS: ' + str(l2NormRMSRook[j])
comparisons = np.array(l2NormRMSNorm) - np.array(l2NormRMSRook)
print 'Norm Mean: ' + str(np.mean(np.array(l2NormRMSNorm)))
print 'Rook Mean: ' + str(np.mean(np.array(l2NormRMSRook)))
plt.plot(comparisons)
plt.show()
plt.clf()
print '- Validation Cell Complete -'
'''


# In[ ]:


'''
import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd*10)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
predictionsList = []
rList = np.arange(500)
np.random.shuffle(rList)
for j in xrange(0,250,1):
    packType = np.random.randint(2)
    selection = rList[j] * 0.0015
    fNT = fNormVal[int(selection*(fNormVal.shape[0])):int((selection+0.25)*(fNormVal.shape[0])),:]
    fRT = fRookVal[int(selection*(fRookVal.shape[0])):int((selection+0.25)*(fRookVal.shape[0])),:]
    l2NormList = []
    if(packType==0):
        for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
        if(np.mean(np.array(l2NormList)) <= 2.044):
            predictionsList.append(1)
        else:
            predictionsList.append(0)
    else:
        for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
        if(np.mean(np.array(l2NormList)) >= 2.044):
            predictionsList.append(1)
        else:
            predictionsList.append(0)
print 'Prediction Accuracy: ' + str(np.mean(np.array(predictionsList)))
plt.plot(np.array(predictionsList))
plt.show()
plt.clf()
print '- Validation Cell Complete -'
'''


# In[ ]:


'''
import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd*10)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
rList = np.arange(500)
np.random.shuffle(rList)
for j in xrange(0,250,1):
    selection = rList[j] * 0.0015
    l2NormList = []
    fNT = fTotalTrain[:fTotalTrain.shape[0]//2][int(selection*(fTotalTrain[:fTotalTrain.shape[0]//2].shape[0])):int((selection+0.25)*(fTotalTrain[:fTotalTrain.shape[0]//2].shape[0])),:]
    for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
    print 'norm RMS: ' + str(l2NormRMSNorm[j])
    selection = rList[j+250] * 0.0015
    l2NormList = []
    fRT = fTotalTrain[fTotalTrain.shape[0]//2:][int(selection*(fTotalTrain[fTotalTrain.shape[0]//2:].shape[0])):int((selection+0.25)*(fTotalTrain[fTotalTrain.shape[0]//2:].shape[0])),:]
    for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
        pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRT[i+1+seqL-(seqL//2):i+seqL+1])**2,axis=1)))
    l2NormRMSRook.append(np.mean(np.array(l2NormList)))
    print 'rook RMS: ' + str(l2NormRMSRook[j])
comparisons = np.array(l2NormRMSNorm) - np.array(l2NormRMSRook)
print 'Norm Mean: ' + str(np.mean(np.array(l2NormRMSNorm)))
print 'Rook Mean: ' + str(np.mean(np.array(l2NormRMSRook)))
plt.plot(comparisons)
plt.show()
plt.clf()
print '- Validation Cell Complete -'
'''


# In[5]:


import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
for i in range(100):
    rollFactor = np.random.randint(-seqL//2,seqL//2)
    newTotalTrain = np.roll(fTotalTrain[:fTotalTrain.shape[0]//2],rollFactor,axis=0)
    newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
    newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
    trainList = np.arange(newTotalTrain.shape[0])
    np.random.shuffle(trainList)
    fNormVal = newTotalTrain[trainList[:trainList.shape[0]//2],:,:]
    fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
    rTotalTrain = np.roll(fTotalTrain,np.random.randint(-seqL//2,seqL//2),axis=0)
    rookTotalTrain = np.roll(fTotalTrain[fTotalTrain.shape[0]//2:],rollFactor,axis=0)
    rookTotalTrain = rookTotalTrain[:(rookTotalTrain.shape[0]//seqL)*seqL,:]
    rookTotalTrain = rookTotalTrain.reshape(-1,seqL,rookTotalTrain.shape[1])
    rookTrainList = np.arange(rookTotalTrain.shape[0])
    np.random.shuffle(rookTrainList)
    fRookVal = rookTotalTrain[rookTrainList[:rookTrainList.shape[0]//2],:,:]
    fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
    selList = np.arange(500)
    np.random.shuffle(selList)
    for j in xrange(0,5,1):
        #selection = j * 0.003
        selection = selList[j] * 0.0015
        l2NormList = []
        
        #fNT = fNormVal[int(((selection*(fNormVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fNormVal.shape[0]))//seqL)*seqL),:]
        fNT = fNormVal
        fNT = np.roll(fNT,np.random.randint(fNT.shape[0]//seqL)*seqL,axis=0)
        
        #fNT = np.roll(fNT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):-1,:] - fNT[i+1+seqL-(seqL//2):(i+seqL+1)-1])**2,axis=1)))
        l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
        #print 'norm RMS: ' + str(l2NormRMSNorm[j])
        selection = selList[j+250] * 0.0015
        l2NormList = []
        
        #fRT = fRookVal[int(((selection*(fRookVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fRookVal.shape[0]))//seqL)*seqL),:]
        fRT = fRookVal
        fRT = np.roll(fRT,np.random.randint(fRT.shape[0]//seqL)*seqL,axis=0)
        
        #fRT = np.roll(fRT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):-1,:] - fRT[i+1+seqL-(seqL//2):(i+seqL+1)-1])**2,axis=1)))
        l2NormRMSRook.append(np.mean(np.array(l2NormList)))
        #print 'rook RMS: ' + str(l2NormRMSRook[j])
l2NormRMSNorm = l2NormRMSNorm[:np.min((len(l2NormRMSNorm),len(l2NormRMSRook)))]
l2NormRMSRook = l2NormRMSRook[:np.min((len(l2NormRMSNorm),len(l2NormRMSRook)))]
comparisons = np.array(l2NormRMSNorm) - np.array(l2NormRMSRook)
plt.plot(comparisons)
print 'Norm Mean: ' + str(np.mean(np.array(l2NormRMSNorm)))
print 'Rook Mean: ' + str(np.mean(np.array(l2NormRMSRook)))
print 'Rook Mean-Based Prob: ' + str((np.mean(comparisons) * -1.0) / ((np.mean(np.array(l2NormRMSNorm))+np.mean(np.array(l2NormRMSRook)))/2.0))
print 'Rook Binary-Based Prob: ' + str(float((comparisons < 0).sum()) / float(comparisons.shape[0]))
plt.show()
plt.clf()
print '- Validation Cell Complete -'


# In[3]:


import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalNonDecodedTrain' + str(int(valInd)) + '.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
for i in range(20):
    rollFactor = np.random.randint(-seqL//2,seqL//2)
    newTotalTrain = np.roll(rTotalTrain[:rTotalTrain.shape[0]//2],rollFactor,axis=0)
    newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
    newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
    trainList = np.arange(newTotalTrain.shape[0])
    np.random.shuffle(trainList)
    fNormVal = newTotalTrain[trainList[:trainList.shape[0]//2],:,:]
    fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
    fRookVal = newTotalTrain[trainList[trainList.shape[0]//2:],:,:]
    fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
    selList = np.arange(500)
    np.random.shuffle(selList)
    for j in xrange(0,25,1):
        #selection = j * 0.003
        selection = selList[j] * 0.0015
        l2NormList = []
        
        #fNT = fNormVal[int(((selection*(fNormVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fNormVal.shape[0]))//seqL)*seqL),:]
        fNT = fNormVal
        fNT = np.roll(fNT,np.random.randint(fNT.shape[0]//seqL)*seqL,axis=0)
        
        for i in range(0,fNT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):-1,:] - fNT[i+1+seqL-(seqL//2):(i+seqL+1)-1])**2,axis=1)))
        l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
        #print 'norm RMS: ' + str(l2NormRMSNorm[j])
        selection = selList[j+250] * 0.0015
        l2NormList = []
        
        #fRT = fRookVal[int(((selection*(fRookVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fRookVal.shape[0]))//seqL)*seqL),:]
        fRT = fRookVal
        fRT = np.roll(fRT,np.random.randint(fRT.shape[0]//seqL)*seqL,axis=0)
        
        
        for i in range(0,fRT.shape[0]-(seqL+1),seqL//2):
            pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):-1,:] - fRT[i+1+seqL-(seqL//2):(i+seqL+1)-1])**2,axis=1)))
        l2NormRMSRook.append(np.mean(np.array(l2NormList)))
        #print 'rook RMS: ' + str(l2NormRMSRook[j])
l2NormRMSNorm = l2NormRMSNorm[:np.min((len(l2NormRMSNorm),len(l2NormRMSRook)))]
l2NormRMSRook = l2NormRMSRook[:np.min((len(l2NormRMSNorm),len(l2NormRMSRook)))]
comparisons = np.array(l2NormRMSNorm) - np.array(l2NormRMSRook)
plt.plot(comparisons)
print 'Norm Mean: ' + str(np.mean(np.array(l2NormRMSNorm)))
print 'Rook Mean: ' + str(np.mean(np.array(l2NormRMSRook)))
print 'Rook Mean-Based Prob: ' + str((np.mean(comparisons) * -1.0) / ((np.mean(np.array(l2NormRMSNorm))+np.mean(np.array(l2NormRMSRook)))/2.0))
print 'Rook Binary-Based Prob: ' + str(float((comparisons < 0).sum()) / float(comparisons.shape[0]))
plt.show()
plt.clf()
print '- Validation Cell Complete -'



# coding: utf-8

# In[1]:


import numpy as np
import cPickle as pickle
valInd = 37 # (0 to 80)
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

fNorm = fNorm[:(fNorm.shape[0]//seqL)*seqL,:]
fRook = fRook[:(fNorm.shape[0]//seqL)*seqL,:]
fNormA = fNorm.reshape(-1,seqL,fNorm.shape[1])
fNormATarget = np.roll(fNorm,-1,axis=0).reshape(-1,seqL,fNorm.shape[1])
fNormB = np.roll(fNorm,-seqL//2,axis=0).reshape(-1,seqL,fNorm.shape[1])
fNormBTarget = np.roll(fNorm,(-seqL//2)-1,axis=0).reshape(-1,seqL,fNorm.shape[1])
shufList = np.arange(fNormA.shape[0])
np.random.shuffle(shufList)
fNormA = fNormA[shufList[:shufList.shape[0]//4]]
fNormATarget = fNormATarget[shufList[:shufList.shape[0]//4]]
fNormB = fNormB[shufList[:shufList.shape[0]//4]]
fNormBTarget = fNormBTarget[shufList[:shufList.shape[0]//4]]

fNormTrain = np.concatenate((fNormA,fNormB),axis=0)
fNormTrain = fNormTrain.reshape(fNormTrain.shape[0]*fNormTrain.shape[1],fNormTrain.shape[2])
fNormTarget = np.concatenate((fNormATarget,fNormBTarget),axis=0)
fNormTarget = fNormTarget.reshape(fNormTarget.shape[0]*fNormTarget.shape[1],fNormTarget.shape[2])

fNormA = fRook.reshape(-1,seqL,fRook.shape[1])
fNormATarget = np.roll(fRook,-1,axis=0).reshape(-1,seqL,fRook.shape[1])
fNormB = np.roll(fRook,-seqL//2,axis=0).reshape(-1,seqL,fRook.shape[1])
fNormBTarget = np.roll(fRook,(-seqL//2)-1,axis=0).reshape(-1,seqL,fRook.shape[1])
shufList = np.arange(fNormA.shape[0])
np.random.shuffle(shufList)
fNormA = fNormA[shufList[:shufList.shape[0]//4]]
fNormATarget = fNormATarget[shufList[:shufList.shape[0]//4]]
fNormB = fNormB[shufList[:shufList.shape[0]//4]]
fNormBTarget = fNormBTarget[shufList[:shufList.shape[0]//4]]

fRookTrain = np.concatenate((fNormA,fNormB),axis=0)
fRookTrain = fRookTrain.reshape(fRookTrain.shape[0]*fRookTrain.shape[1],fRookTrain.shape[2])
fRookTarget = np.concatenate((fNormATarget,fNormBTarget),axis=0)
fRookTarget = fRookTarget.reshape(fRookTarget.shape[0]*fRookTarget.shape[1],fRookTarget.shape[2])

fTotalTrain = np.concatenate((fNormTrain,fRookTrain),axis=0)
fTotalTarget = np.concatenate((fNormTarget,fRookTarget),axis=0)
del valMask, fNorm, fRook, fNormTrain, fRookTrain, fNormVal, fRookVal, fNormA, fNormATarget, fNormB, fNormBTarget, fNormTarget, fRookTarget
print 'total features train shape: ' + str(fTotalTrain.shape)
print 'total features target shape: ' + str(fTotalTarget.shape)
#print 'norm features val shape: ' + str(fNormVal.shape)
#print 'rook features val shape: ' + str(fRookVal.shape)
print '- Feature Import Cell Complete -'


# In[2]:


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


# In[3]:


for i in range(200):
    rI = np.random.randint(fTotalTrain.shape[0]//seqL)*seqL
    #xTrain = np.roll(fTotalTrain,-rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    #yTrain = np.roll(fTotalTrain,-(rI+1),axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    #batchSize = np.random.randint(1,4)
    #model.fit(xTrain,yTrain,batch_size=batchSize,epochs=15)
    xTrain = np.roll(fTotalTrain,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    yTrain = np.roll(fTotalTarget,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    model.fit(xTrain,yTrain,batch_size=np.random.randint(1,5),shuffle=False,epochs=10)
model.save('modelTotalTrainRandomized.h5')
del model
print '- Training Cell Complete -'


# In[4]:


import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalTrainRandomized.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
for i in range(50):
    #rollFactor = np.random.randint(-seqL//2,seqL//2)
    #newTotalTrain = np.roll(fTotalTrain[:fTotalTrain.shape[0]//2],rollFactor,axis=0)
    newTotalTrain = fTotalTrain[:fTotalTrain.shape[0]//4]
    newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
    newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
    newTotalTarget = fTotalTarget[:fTotalTarget.shape[0]//4]
    newTotalTarget = newTotalTarget[:(newTotalTarget.shape[0]//seqL)*seqL,:]
    newTotalTarget = newTotalTarget.reshape(-1,seqL,newTotalTarget.shape[1])
    trainList = np.arange(newTotalTrain.shape[0])
    np.random.shuffle(trainList)
    fNormVal = newTotalTrain[trainList[:trainList.shape[0]],:,:]
    fNormTar = newTotalTarget[trainList[:trainList.shape[0]],:,:]
    fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
    fNormTar = fNormTar.reshape(fNormTar.shape[0]*fNormTar.shape[1],fNormTar.shape[2])
    #rTotalTrain = np.roll(fTotalTrain,np.random.randint(-seqL//2,seqL//2),axis=0)
    #rookTotalTrain = np.roll(fTotalTrain[fTotalTrain.shape[0]//2:],rollFactor,axis=0)
    rookTotalTrain = fTotalTrain[fTotalTrain.shape[0]//2:(3*fTotalTrain.shape[0])//4]
    rookTotalTrain = rookTotalTrain[:(rookTotalTrain.shape[0]//seqL)*seqL,:]
    rookTotalTrain = rookTotalTrain.reshape(-1,seqL,rookTotalTrain.shape[1])
    rookTotalTarget = fTotalTarget[fTotalTarget.shape[0]//2:(3*fTotalTarget.shape[0])//4]
    rookTotalTarget = rookTotalTarget[:(rookTotalTarget.shape[0]//seqL)*seqL,:]
    rookTotalTarget = rookTotalTarget.reshape(-1,seqL,rookTotalTarget.shape[1])
    rookTrainList = np.arange(rookTotalTrain.shape[0])
    np.random.shuffle(rookTrainList)
    fRookVal = rookTotalTrain[rookTrainList[:rookTrainList.shape[0]],:,:]
    fRookTar = rookTotalTarget[rookTrainList[:rookTrainList.shape[0]],:,:]
    fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
    fRookTar = fRookTar.reshape(fRookTar.shape[0]*fRookTar.shape[1],fRookTar.shape[2])
    #selList = np.arange(500)
    #np.random.shuffle(selList)
    for j in xrange(0,4,1):
        #selection = j * 0.003
        #selection = selList[j] * 0.0015
        l2NormList = []
        
        #fNT = fNormVal[int(((selection*(fNormVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fNormVal.shape[0]))//seqL)*seqL),:]
        myRoll = np.random.randint(fNormVal.shape[0]//seqL)*seqL
        fNT = np.roll(fNormVal,myRoll,axis=0)
        fNTar = np.roll(fNormTar,myRoll,axis=0)
        
        #fNT = np.roll(fNT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fNT.shape[0]-(seqL+1),seqL):
            pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
        l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
        #print 'norm RMS: ' + str(l2NormRMSNorm[j])
        #selection = selList[j+250] * 0.0015
        l2NormList = []
        
        #fRT = fRookVal[int(((selection*(fRookVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fRookVal.shape[0]))//seqL)*seqL),:]
        myRoll = np.random.randint(fRookVal.shape[0]//seqL)*seqL
        fRT = np.roll(fRookVal,myRoll,axis=0)
        fRTar = np.roll(fRookTar,myRoll,axis=0)
        
        #fRT = np.roll(fRT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fRT.shape[0]-(seqL+1),seqL):
            pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
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


# In[5]:


import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalTrainRandomized.h5')
l2NormRMSNorm = []
l2NormRMSRook = []
for i in range(50):
    #rollFactor = np.random.randint(-seqL//2,seqL//2)
    #newTotalTrain = np.roll(fTotalTrain[:fTotalTrain.shape[0]//2],rollFactor,axis=0)
    newTotalTrain = fTotalTrain[:fTotalTrain.shape[0]//4]
    newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
    newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
    newTotalTarget = fTotalTarget[:fTotalTarget.shape[0]//4]
    newTotalTarget = newTotalTarget[:(newTotalTarget.shape[0]//seqL)*seqL,:]
    newTotalTarget = newTotalTarget.reshape(-1,seqL,newTotalTarget.shape[1])
    trainList = np.arange(newTotalTrain.shape[0])
    np.random.shuffle(trainList)
    fNormVal = newTotalTrain[trainList[:trainList.shape[0]],:,:]
    fNormTar = newTotalTarget[trainList[:trainList.shape[0]],:,:]
    fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
    fNormTar = fNormTar.reshape(fNormTar.shape[0]*fNormTar.shape[1],fNormTar.shape[2])
    #rTotalTrain = np.roll(fTotalTrain,np.random.randint(-seqL//2,seqL//2),axis=0)
    #rookTotalTrain = np.roll(fTotalTrain[fTotalTrain.shape[0]//2:],rollFactor,axis=0)
    rookTotalTrain = fTotalTrain[fTotalTrain.shape[0]//4:fTotalTrain.shape[0]//2]
    rookTotalTrain = rookTotalTrain[:(rookTotalTrain.shape[0]//seqL)*seqL,:]
    rookTotalTrain = rookTotalTrain.reshape(-1,seqL,rookTotalTrain.shape[1])
    rookTotalTarget = fTotalTarget[fTotalTarget.shape[0]//4:fTotalTarget.shape[0]//2]
    rookTotalTarget = rookTotalTarget[:(rookTotalTarget.shape[0]//seqL)*seqL,:]
    rookTotalTarget = rookTotalTarget.reshape(-1,seqL,rookTotalTarget.shape[1])
    rookTrainList = np.arange(rookTotalTrain.shape[0])
    np.random.shuffle(rookTrainList)
    fRookVal = rookTotalTrain[rookTrainList[:rookTrainList.shape[0]],:,:]
    fRookTar = rookTotalTarget[rookTrainList[:rookTrainList.shape[0]],:,:]
    fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
    fRookTar = fRookTar.reshape(fRookTar.shape[0]*fRookTar.shape[1],fRookTar.shape[2])
    #selList = np.arange(500)
    #np.random.shuffle(selList)
    for j in xrange(0,4,1):
        #selection = j * 0.003
        #selection = selList[j] * 0.0015
        l2NormList = []
        
        #fNT = fNormVal[int(((selection*(fNormVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fNormVal.shape[0]))//seqL)*seqL),:]
        myRoll = np.random.randint(fNormVal.shape[0]//seqL)*seqL
        fNT = np.roll(fNormVal,myRoll,axis=0)
        fNTar = np.roll(fNormTar,myRoll,axis=0)
        
        #fNT = np.roll(fNT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fNT.shape[0]-(seqL+1),seqL):
            pred = model.predict(fNT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fNTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
        l2NormRMSNorm.append(np.mean(np.array(l2NormList)))
        #print 'norm RMS: ' + str(l2NormRMSNorm[j])
        #selection = selList[j+250] * 0.0015
        l2NormList = []
        
        #fRT = fRookVal[int(((selection*(fRookVal.shape[0]))//seqL)*seqL):int((((selection+0.25)*(fRookVal.shape[0]))//seqL)*seqL),:]
        myRoll = np.random.randint(fRookVal.shape[0]//seqL)*seqL
        fRT = np.roll(fRookVal,myRoll,axis=0)
        fRTar = np.roll(fRookTar,myRoll,axis=0)
        
        #fRT = np.roll(fRT,np.random.randint(-seqL//2,seqL//2),axis=0)
        for i in range(0,fRT.shape[0]-(seqL+1),seqL):
            pred = model.predict(fRT[i:i+seqL,:][np.newaxis,:,:])[0]
            l2NormList.append(np.sqrt(np.sum((pred[(-seqL//2):,:] - fRTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
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


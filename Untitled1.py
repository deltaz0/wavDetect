
# coding: utf-8

# In[1]:


import numpy as np
import cPickle as pickle
valInd = 37 # (0 to 80)
seqL = 200
charInd = 'A'
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
pickle.dump(fTotalTrain,open('fTotalTrain' + charInd + '.pkl', 'wb'))
pickle.dump(fTotalTarget,open('fTotalTarget' + charInd + '.pkl', 'wb'))
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


for i in range(8):
    rI = np.random.randint(fTotalTrain.shape[0]//seqL)*seqL
    #xTrain = np.roll(fTotalTrain,-rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    #yTrain = np.roll(fTotalTrain,-(rI+1),axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    #batchSize = np.random.randint(1,4)
    #model.fit(xTrain,yTrain,batch_size=batchSize,epochs=15)
    xTrain = np.roll(fTotalTrain,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    yTrain = np.roll(fTotalTarget,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
    model.fit(xTrain,yTrain,batch_size=np.random.randint(1,5),shuffle=False,epochs=10)
model.save('modelTotalTrainRandomized' + charInd + '.h5')
del model
print '- Training Cell Complete -'


# In[4]:


import matplotlib.pyplot as plt
from keras.models import Model, load_model
model = load_model('modelTotalTrainRandomized' + charInd + '.h5')

l2NormList = []
l2NormBList = []
l2RookList = []

for j in range(4):
    newTotalTrain = fTotalTrain[:fTotalTrain.shape[0]//2]
    newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
    newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
    newTotalTarget = fTotalTarget[:fTotalTarget.shape[0]//2]
    newTotalTarget = newTotalTarget[:(newTotalTarget.shape[0]//seqL)*seqL,:]
    newTotalTarget = newTotalTarget.reshape(-1,seqL,newTotalTarget.shape[1])
    trainList = np.arange(newTotalTrain.shape[0]//2)
    np.random.shuffle(trainList)
    fNormVal = np.concatenate((newTotalTrain[trainList[:trainList.shape[0]//2],:,:],newTotalTrain[trainList[:trainList.shape[0]//2]+trainList.shape[0],:,:]),axis=0)
    fNormTar = np.concatenate((newTotalTarget[trainList[:trainList.shape[0]//2],:,:],newTotalTarget[trainList[:trainList.shape[0]//2]+trainList.shape[0],:,:]),axis=0)
    fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
    fNormTar = fNormTar.reshape(fNormTar.shape[0]*fNormTar.shape[1],fNormTar.shape[2])

    fNormBVal = np.concatenate((newTotalTrain[trainList[trainList.shape[0]//2:],:,:],newTotalTrain[trainList[trainList.shape[0]//2:]+trainList.shape[0],:,:]),axis=0)
    fNormBTar = np.concatenate((newTotalTarget[trainList[trainList.shape[0]//2:],:,:],newTotalTarget[trainList[trainList.shape[0]//2:]+trainList.shape[0],:,:]),axis=0)
    fNormBVal = fNormBVal.reshape(fNormBVal.shape[0]*fNormBVal.shape[1],fNormBVal.shape[2])
    fNormBTar = fNormBTar.reshape(fNormBTar.shape[0]*fNormBTar.shape[1],fNormBTar.shape[2])

    if(fNormVal.shape[0] < fNormBVal.shape[0]):
        fNormBVal = fNormBVal[:fNormVal.shape[0]]
        fNormBTar = fNormBTar[:fNormTar.shape[0]]
    else:
        fNormVal = fNormVal[:fNormBVal.shape[0]]
        fNormTar = fNormTar[:fNormBTar.shape[0]]

    rookTotalTrain = fTotalTrain[fTotalTrain.shape[0]//2:]
    rookTotalTrain = rookTotalTrain[:(rookTotalTrain.shape[0]//seqL)*seqL,:]
    rookTotalTrain = rookTotalTrain.reshape(-1,seqL,rookTotalTrain.shape[1])
    rookTotalTarget = fTotalTarget[fTotalTarget.shape[0]//2:]
    rookTotalTarget = rookTotalTarget[:(rookTotalTarget.shape[0]//seqL)*seqL,:]
    rookTotalTarget = rookTotalTarget.reshape(-1,seqL,rookTotalTarget.shape[1])
    rookTrainList = np.arange(rookTotalTrain.shape[0]//2)
    np.random.shuffle(rookTrainList)
    fRookVal = np.concatenate((rookTotalTrain[rookTrainList[:rookTrainList.shape[0]//2],:,:],rookTotalTrain[rookTrainList[:rookTrainList.shape[0]//2]+rookTrainList.shape[0],:,:]),axis=0)
    fRookTar = np.concatenate((rookTotalTarget[rookTrainList[:rookTrainList.shape[0]//2],:,:],rookTotalTarget[rookTrainList[:rookTrainList.shape[0]//2]+rookTrainList.shape[0],:,:]),axis=0)
    fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
    fRookTar = fRookTar.reshape(fRookTar.shape[0]*fRookTar.shape[1],fRookTar.shape[2])

    if(fRookVal.shape[0] < fNormBVal.shape[0]):
        fNormBVal = fNormBVal[:fRookVal.shape[0]]
        fNormBTar = fNormBTar[:fRookTar.shape[0]]
        fNormVal = fNormVal[:fRookVal.shape[0]]
        fNormTar = fNormTar[:fRookTar.shape[0]]
    else:
        fRookVal = fRookVal[:fNormBVal.shape[0]]
        fRookTar = fRookTar[:fNormBTar.shape[0]]

    for i in range(0,fNormVal.shape[0]-(seqL+1),seqL):
        normPred = model.predict(fNormVal[i:i+seqL][np.newaxis,:,:])[0]
        l2NormList.append(np.sqrt(np.sum((normPred[(-seqL//2):,:] - fNormTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
        normBPred = model.predict(fNormBVal[i:i+seqL][np.newaxis,:,:])[0]
        l2NormBList.append(np.sqrt(np.sum((normBPred[(-seqL//2):,:] - fNormBTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))
        rookPred = model.predict(fRookVal[i:i+seqL][np.newaxis,:,:])[0]
        l2RookList.append(np.sqrt(np.sum((rookPred[(-seqL//2):,:] - fRookTar[i+seqL-(seqL//2):(i+seqL)])**2,axis=1)))

normBNormComp = []
rookNormComp = []
print 'Norm Mean: ' + str(np.mean(np.array(l2NormList)))
print 'NormB Mean: ' + str(np.mean(np.array(l2NormBList)))
print 'Rook Mean: ' + str(np.mean(np.array(l2RookList)))
meanAvg = (np.mean(np.array(l2NormList)) + np.mean(np.array(l2NormBList)) + np.mean(np.array(l2RookList))) / 3.0
print 'Avg Mean: ' + str(meanAvg)
print 'Norm StdDev: ' + str(np.std(np.array(l2NormList)))
print 'NormB StdDev: ' + str(np.std(np.array(l2NormBList)))
print 'Rook StdDev: ' + str(np.std(np.array(l2RookList)))
print 'Norm RMS: ' + str(np.sqrt(np.mean(np.array(l2NormList)**2)))
print 'NormB RMS: ' + str(np.sqrt(np.mean(np.array(l2NormBList)**2)))
print 'Rook RMS: ' + str(np.sqrt(np.mean(np.array(l2RookList)**2)))
totalRMS = np.sqrt(np.mean(np.concatenate((np.concatenate((np.array(l2NormList)**2,np.array(l2NormBList)**2)),np.array(l2RookList)**2))))
print 'Total RMS: ' + str(totalRMS)
print 'NormB vs Norm Scaled Mean Comparison (should be low, around 0): ' + str((np.mean(np.array(l2NormBList)) - np.mean(np.array(l2NormList))) / meanAvg)
print 'Rook vs Norm Scaled Mean Comparison (should be higher): ' + str((np.mean(np.array(l2RookList)) - np.mean(np.array(l2NormList))) / meanAvg)
print 'NormB vs Norm Scaled RMS Comparison (should be low, around 0): ' + str((np.sqrt(np.mean(np.array(l2NormBList)**2)) - np.sqrt(np.mean(np.array(l2NormList)**2))) / totalRMS)
print 'Rook vs Norm Scaled RMS Comparison (should be higher): ' + str((np.sqrt(np.mean(np.array(l2RookList)**2)) - np.sqrt(np.mean(np.array(l2NormList)**2))) / totalRMS)
print 'NormB vs Norm RMS of Comparisons (should be low, around 0): ' + str(np.sqrt(np.abs(np.mean(np.array(l2NormBList)**2) - np.mean(np.array(l2NormList)**2))) / totalRMS)
print 'Rook vs Norm RMS of Comparisons (should be higher): ' + str(np.sqrt(np.abs(np.mean(np.array(l2RookList)**2) - np.mean(np.array(l2NormList)**2))) / totalRMS)
print 'NormB vs Norm Binary Comparisons (should be low, around 0): ' + str(float(((np.array(l2NormBList) - np.array(l2NormList)) >0).sum()) / float((np.array(l2NormBList) - np.array(l2NormList)).shape[0]))
print 'Rook vs Norm Binary Comparisons (should be higher): ' + str(float(((np.array(l2RookList) - np.array(l2NormList)) >0).sum()) / float((np.array(l2RookList) - np.array(l2NormList)).shape[0]))
for i in range(int(len(l2NormList)-((len(l2NormList)//4)+1))):
    normBNormComp.append((np.mean(np.array(l2NormBList[i:i+len(l2NormBList)//4])) - np.mean(np.array(l2NormList[i:i+len(l2NormList)//4]))) / meanAvg)
    rookNormComp.append((np.mean(np.array(l2RookList[i:i+len(l2RookList)//4])) - np.mean(np.array(l2NormList[i:i+len(l2NormList)//4]))) / meanAvg)
plt.plot(normBNormComp,label='Norm v Norm')
plt.plot(rookNormComp,label='Rook v Norm')
plt.legend()
plt.show()
plt.savefig('predictionErrorComps' + charInd + '.png')
plt.clf()
print '- Validation Cell Complete -'


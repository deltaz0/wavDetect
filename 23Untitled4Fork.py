import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pickle as pickle
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv1D, TimeDistributed
import matplotlib.pyplot as plt
import os

# totalLen = 80000
totalLen = 161840
numFeatures = 32
minPeriod = 10
maxPeriod = 200
# filterSize = 32
filterSize = 64
modelVersion = 'rollhalfseqbase32iters'
resultsVersion = 'A'
permuteStep = 8
seqL = 200
trainIters = 32
batchSize = 64

faketoy = False
permuteType = 'rand' # rand or sin
decode = True
mask = 0 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv

fullRoll = True

rollAmt = 1
if fullRoll:
    rollAmt = seqL//2

for runType in [0, 1, 2, 3, 4, 5, 6, 7]:
    if runType == 0:
        faketoy = False
        decode = True
        mask = 0 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 1:
        faketoy = False
        decode = True
        mask = 1 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 2:
        faketoy = False
        decode = True
        mask = 2 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 3:
        faketoy = False
        decode = False
        mask = 0 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 4:
        faketoy = False
        decode = False
        mask = 1 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 5:
        faketoy = False
        decode = False
        mask = 2 # 0 for nomask, 1 for maskbase, 2 for maskbasederiv
    elif runType == 6:
        faketoy = True
        permuteType = 'rand' # rand or sin
    elif runType == 7:
        faketoy = True
        permuteType = 'sin' # rand or sin
    fileSuffix = ''
    if faketoy == True:
        if permuteType == 'rand':
            fileSuffix = 'FakeToyPermuteRand'
        else:
            fileSuffix = 'FakeToyPermuteSin'
    else:
        if decode == True:
            if mask == 2:            
                fileSuffix = 'New23DecMaskDeriv'
            elif mask == 1:
                fileSuffix = 'New23DecMaskBase'
            else:
                fileSuffix = 'New23DecNoMask'
        else:
            if mask == 2:
                fileSuffix = 'New23NDMaskDeriv'
            elif mask == 1:
                fileSuffix = 'New23NDMaskBase'
            else:
                fileSuffix = 'New23NDNoMask'
    if not os.path.exists('results/' + modelVersion + '-' + fileSuffix + '-' + resultsVersion + '/'):
        os.mkdir('results/' + modelVersion + '-' + fileSuffix + '-' + resultsVersion)
    resultsFileName = modelVersion + '-' + fileSuffix + '-' + resultsVersion + '/'
    if not os.path.exists('models/' + modelVersion + '-' + fileSuffix + '/'):
        os.mkdir('models/' + modelVersion + '-' + fileSuffix)

    if faketoy:
        x = np.zeros((int(totalLen*2),numFeatures))
        for i in range(numFeatures):
        #per = float(np.random.randint(minPeriod,maxPeriod))
            per = float(np.random.rand()*(maxPeriod-minPeriod)+minPeriod)
            amp = float(np.random.rand())
            midpoint = float(np.random.rand() * (2*(1.0-amp)) - (1.0 - amp))
            a = np.arange(0,2*np.pi*(float(int(totalLen*2))/per),(2*np.pi)/per)[:int(totalLen*2)]
            r = np.random.randint(3)
            if(r==0):
                x[:,i] = np.sin(a) * amp + midpoint
            elif(r==1):
                x[:,i] = signal.sawtooth(a) * amp + midpoint
            else:
                x[:,i] = signal.square(a) * amp + midpoint
        x = np.sum(x/x.shape[1],axis=1)[:,np.newaxis]
        # plt.plot(x)
        # plt.show()
        # plt.clf()
        y = np.copy(x[x.shape[0]//2:])
        x = x[:x.shape[0]//2]
        if permuteType == 'rand':
            y[np.arange(0,y.shape[0],permuteStep),
            np.random.randint(y.shape[1],size=y.shape[0]//permuteStep)] = \
                np.random.rand(y.shape[0]//permuteStep)
        elif permuteType == 'sin':
            y[np.arange(0,y.shape[0],permuteStep),
            np.random.randint(y.shape[1],size=y.shape[0]//permuteStep)] = \
                np.sin(np.arange(0,y.shape[0]//permuteStep) * \
                    (np.random.rand()*1500+1000)*np.pi/(y.shape[0]//permuteStep))
        else:
            raise Exception('need to provide permute type')
        # plt.plot(y)
        # plt.show()
        # plt.clf()
        # fNormX = np.copy(x[(totalLen//2):])
        # fNorm = x[:(totalLen//2)]
        # fRookY = np.copy(y[(totalLen//2):])
        # fRook = y[:(totalLen//2)]
        fNorm = x
        fRook = y
    else:
        pklFileNorm = open('../ncoderdata/featuresNormal' + fileSuffix + '.pkl', 'rb')
        pklFileRook = open('../ncoderdata/featuresRook' + fileSuffix + '.pkl', 'rb')            
        fNorm = pickle.load(pklFileNorm)
        fRook = pickle.load(pklFileRook)
        pklFileNorm.close()
        pklFileRook.close()
        # fNormOne = np.concatenate((fNorm[:fNorm.shape[0]//4],fNorm[(3*fNorm.shape[0])//4:]),axis=0)
        # fNormTwo = np.copy(fNorm[fNorm.shape[0]//4:(3*fNorm.shape[0])//4])
        # fRookOne = np.copy(fRook[:fRook.shape[0]//2])
        # fRookTwo = np.copy(fRook[fRook.shape[0]//2:])
        # fNorm = fNormOne
        # fNormX = fNormTwo
        # fRook = fRookOne
        # fRookY = fRookTwo
    if(fNorm.shape[0] > fRook.shape[0]):
        fNorm = fNorm[:fRook.shape[0]]
    else:
        fRook = fRook[:fNorm.shape[0]]
    # if(fNormX.shape[0] > fRookY.shape[0]):
    #     fNormX = fNormX[:fRookY.shape[0]]
    # else:
    #     fRookY = fRookY[:fNormX.shape[0]]


    dummyIndList = np.arange(fNorm.shape[0])
    div = np.random.rand(1)[0]
    dummyIndList = np.roll(dummyIndList,-int(div * dummyIndList.shape[0]))
    # dummyIndList3 = np.arange(fNormX.shape[0])
    # div3 = np.random.rand(1)[0]
    # dummyIndList3 = np.roll(dummyIndList3,-int(div3 * dummyIndList3.shape[0]))
    fNormOne = fNorm[dummyIndList[:dummyIndList.shape[0]//2]]
    fNormTwo = fNorm[dummyIndList[dummyIndList.shape[0]//2:]]
    # fNormThree = fNormX[dummyIndList3[:dummyIndList3.shape[0]//2]]
    # fNormFour = fNormX[dummyIndList3[dummyIndList3.shape[0]//2:]]
    dummyIndList = np.roll(dummyIndList,-int(dummyIndList.shape[0]//4))
    # dummyIndList3 = np.roll(dummyIndList3,-int(dummyIndList3.shape[0]//4))
    fRookOne = fRook[dummyIndList[:dummyIndList.shape[0]//2]]
    fRookTwo = fRook[dummyIndList[dummyIndList.shape[0]//2:]]
    # fRookThree = fRookY[dummyIndList3[:dummyIndList3.shape[0]//2]]
    # fRookFour = fRookY[dummyIndList3[dummyIndList3.shape[0]//2:]]
    #fNormOne = np.concatenate((fNorm[:fNorm.shape[0]//4],fNorm[(3*fNorm.shape[0])//4:]),axis=0)
    #fNormTwo = np.copy(fNorm[fNorm.shape[0]//4:(3*fNorm.shape[0])//4])
    #fRookOne = np.copy(fRook[:fRook.shape[0]//2])
    #fRookTwo = np.copy(fRook[fRook.shape[0]//2:])
    fNorm = fNormOne[:(fNormOne.shape[0]//seqL)*seqL,:]
    fNormTwo = fNormTwo[:(fNormTwo.shape[0]//seqL)*seqL,:]
    # fNormThree = fNormThree[:(fNormThree.shape[0]//seqL)*seqL,:]
    # fNormFour = fNormFour[:(fNormFour.shape[0]//seqL)*seqL,:]
    fRook = fRookOne[:(fRookOne.shape[0]//seqL)*seqL,:]
    fRookTwo = fRookTwo[:(fRookTwo.shape[0]//seqL)*seqL,:]
    # fRookThree = fRookThree[:(fRookThree.shape[0]//seqL)*seqL,:]
    # fRookFour = fRookFour[:(fRookFour.shape[0]//seqL)*seqL,:]

    l2NormListHolder    =   []
    l2RookListHolder    =   []
    l2Norm3OnNNHolder   =   []
    l2Norm4OnNNHolder   =   []
    l2Rook3OnNNHolder   =   []
    l2Norm3OnRRHolder   =   []
    l2Norm4OnRRHolder   =   []
    l2Rook3OnRRHolder   =   []
    l2NormBListHolder   =   []
    l2RookBListHolder   =   []
    l2NormAltListHolder =   []
    l2RookAltListHolder =   []
    for topInd in range(4):
        shufList = np.arange(fNorm.shape[0]//seqL)
        np.random.shuffle(shufList)
        shufListB = np.arange(fNormTwo.shape[0]//seqL)
        np.random.shuffle(shufListB)
        # shufListB3 = np.arange(fNormThree.shape[0]//seqL)
        # np.random.shuffle(shufListB3)
        # shufListB4 = np.arange(fNormFour.shape[0]//seqL)
        # np.random.shuffle(shufListB4)
        shufListR = np.arange(fRook.shape[0]//seqL)
        np.random.shuffle(shufListR)
        shufListC = np.arange(fRookTwo.shape[0]//seqL)
        np.random.shuffle(shufListC)
        # shufListC3 = np.arange(fRookThree.shape[0]//seqL)
        # np.random.shuffle(shufListC3)
        # shufListC4 = np.arange(fRookFour.shape[0]//seqL)
        # np.random.shuffle(shufListC4)
        #bShuf = []
        #for i in range(4):
        #    bShuf.append((np.arange(4)[np.arange(4)!=i])[np.random.randint(3)])
        for subInd in range(4):
            # fNormA3 = fNormThree.reshape(-1,seqL,fNormThree.shape[1])
            # fNormA3Target = np.roll(fNormThree,-1,axis=0).reshape(-1,seqL,fNormThree.shape[1])
            # fNormB3 = np.roll(fNormThree,-seqL//2,axis=0).reshape(-1,seqL,fNormThree.shape[1])
            # fNormB3Target = np.roll(fNormThree,(-seqL//2)-1,axis=0).reshape(-1,seqL,fNormThree.shape[1])
            # fNormAB3 = fNormA3[shufListB3[(subInd*shufListB3.shape[0])//4:((subInd+1)*shufListB3.shape[0])//4]]
            # fNormATargetB3 = fNormA3Target[shufListB3[(subInd*shufListB3.shape[0])//4:((subInd+1)*shufListB3.shape[0])//4]]
            # fNormBB3 = fNormB3[shufListB3[(subInd*shufListB3.shape[0])//4:((subInd+1)*shufListB3.shape[0])//4]]
            # fNormBTargetB3 = fNormB3Target[shufListB3[(subInd*shufListB3.shape[0])//4:((subInd+1)*shufListB3.shape[0])//4]]
            # fNormTrainB3 = np.concatenate((fNormAB3,fNormBB3),axis=0)
            # fNormTrainB3 = fNormTrainB3.reshape(fNormTrainB3.shape[0]*fNormTrainB3.shape[1],fNormTrainB3.shape[2])
            # fNormTargetB3 = np.concatenate((fNormATargetB3,fNormBTargetB3),axis=0)
            # fNormTargetB3 = fNormTargetB3.reshape(fNormTargetB3.shape[0]*fNormTargetB3.shape[1],fNormTargetB3.shape[2])
            
            # fNormA3 = fRookThree.reshape(-1,seqL,fRookThree.shape[1])
            # fNormA3Target = np.roll(fRookThree,-1,axis=0).reshape(-1,seqL,fRookThree.shape[1])
            # fNormB3 = np.roll(fRookThree,-seqL//2,axis=0).reshape(-1,seqL,fRookThree.shape[1])
            # fNormB3Target = np.roll(fRookThree,(-seqL//2)-1,axis=0).reshape(-1,seqL,fRookThree.shape[1])
            # fNormAB3 = fNormA3[shufListC3[(subInd*shufListC3.shape[0])//4:((subInd+1)*shufListC3.shape[0])//4]]
            # fNormATargetB3 = fNormA3Target[shufListC3[(subInd*shufListC3.shape[0])//4:((subInd+1)*shufListC3.shape[0])//4]]
            # fNormBB3 = fNormB3[shufListC3[(subInd*shufListC3.shape[0])//4:((subInd+1)*shufListC3.shape[0])//4]]
            # fNormBTargetB3 = fNormB3Target[shufListC3[(subInd*shufListC3.shape[0])//4:((subInd+1)*shufListC3.shape[0])//4]]
            # fRookTrainB3 = np.concatenate((fNormAB3,fNormBB3),axis=0)
            # fRookTrainB3 = fRookTrainB3.reshape(fRookTrainB3.shape[0]*fRookTrainB3.shape[1],fRookTrainB3.shape[2])
            # fRookTargetB3 = np.concatenate((fNormATargetB3,fNormBTargetB3),axis=0)
            # fRookTargetB3 = fRookTargetB3.reshape(fRookTargetB3.shape[0]*fRookTargetB3.shape[1],fRookTargetB3.shape[2])
            
            # fNormA4 = fNormFour.reshape(-1,seqL,fNormFour.shape[1])
            # fNormA4Target = np.roll(fNormFour,-1,axis=0).reshape(-1,seqL,fNormFour.shape[1])
            # fNormB4 = np.roll(fNormFour,-seqL//2,axis=0).reshape(-1,seqL,fNormFour.shape[1])
            # fNormB4Target = np.roll(fNormFour,(-seqL//2)-1,axis=0).reshape(-1,seqL,fNormFour.shape[1])
            # fNormAB4 = fNormA4[shufListB4[(subInd*shufListB4.shape[0])//4:((subInd+1)*shufListB4.shape[0])//4]]
            # fNormATargetB4 = fNormA4Target[shufListB4[(subInd*shufListB4.shape[0])//4:((subInd+1)*shufListB4.shape[0])//4]]
            # fNormBB4 = fNormB4[shufListB4[(subInd*shufListB4.shape[0])//4:((subInd+1)*shufListB4.shape[0])//4]]
            # fNormBTargetB4 = fNormB4Target[shufListB4[(subInd*shufListB4.shape[0])//4:((subInd+1)*shufListB4.shape[0])//4]]
            # fNormTrainB4 = np.concatenate((fNormAB4,fNormBB4),axis=0)
            # fNormTrainB4 = fNormTrainB4.reshape(fNormTrainB4.shape[0]*fNormTrainB4.shape[1],fNormTrainB4.shape[2])
            # fNormTargetB4 = np.concatenate((fNormATargetB4,fNormBTargetB4),axis=0)
            # fNormTargetB4 = fNormTargetB4.reshape(fNormTargetB4.shape[0]*fNormTargetB4.shape[1],fNormTargetB4.shape[2])
            
            # fNormA4 = fRookFour.reshape(-1,seqL,fRookFour.shape[1])
            # fNormA4Target = np.roll(fRookFour,-1,axis=0).reshape(-1,seqL,fRookFour.shape[1])
            # fNormB4 = np.roll(fRookFour,-seqL//2,axis=0).reshape(-1,seqL,fRookFour.shape[1])
            # fNormB4Target = np.roll(fRookFour,(-seqL//2)-1,axis=0).reshape(-1,seqL,fRookFour.shape[1])
            # fNormAB4 = fNormA4[shufListC4[(subInd*shufListC4.shape[0])//4:((subInd+1)*shufListC4.shape[0])//4]]
            # fNormATargetB4 = fNormA4Target[shufListC4[(subInd*shufListC4.shape[0])//4:((subInd+1)*shufListC4.shape[0])//4]]
            # fNormBB4 = fNormB4[shufListC4[(subInd*shufListC4.shape[0])//4:((subInd+1)*shufListC4.shape[0])//4]]
            # fNormBTargetB4 = fNormB4Target[shufListC4[(subInd*shufListC4.shape[0])//4:((subInd+1)*shufListC4.shape[0])//4]]
            # fRookTrainB4 = np.concatenate((fNormAB4,fNormBB4),axis=0)
            # fRookTrainB4 = fRookTrainB4.reshape(fRookTrainB4.shape[0]*fRookTrainB4.shape[1],fRookTrainB4.shape[2])
            # fRookTargetB4 = np.concatenate((fNormATargetB4,fNormBTargetB4),axis=0)
            # fRookTargetB4 = fRookTargetB4.reshape(fRookTargetB4.shape[0]*fRookTargetB4.shape[1],fRookTargetB4.shape[2])
            
            #fNormAB = fNormA[shufListB[(bShuf[subInd]*shufListB.shape[0])//4:(bShuf[(subInd+1)%4]*shufListB.shape[0])//4]]
            #fNormATargetB = fNormATarget[shufListB[(bShuf[subInd]*shufListB.shape[0])//4:(bShuf[(subInd+1)%4]*shufListB.shape[0])//4]]
            #fNormBB = fNormB[shufListB[(bShuf[subInd]*shufListB.shape[0])//4:(bShuf[(subInd+1)%4]*shufListB.shape[0])//4]]
            #fNormBTargetB = fNormBTarget[shufListB[(bShuf[subInd]*shufListB.shape[0])//4:(bShuf[(subInd+1)%4]*shufListB.shape[0])//4]]
                    
            fNormA = fNormTwo.reshape(-1,seqL,fNormTwo.shape[1])
            fNormATarget = np.roll(fNormTwo,-rollAmt,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
            fNormB = np.roll(fNormTwo,-seqL//2,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
            fNormBTarget = np.roll(fNormTwo,(-seqL//2)-rollAmt,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
            fNormAB = fNormA[shufListB[(subInd*shufListB.shape[0])//4:((subInd+1)*shufListB.shape[0])//4]]
            fNormATargetB = fNormATarget[shufListB[(subInd*shufListB.shape[0])//4:((subInd+1)*shufListB.shape[0])//4]]
            fNormBB = fNormB[shufListB[(subInd*shufListB.shape[0])//4:((subInd+1)*shufListB.shape[0])//4]]
            fNormBTargetB = fNormBTarget[shufListB[(subInd*shufListB.shape[0])//4:((subInd+1)*shufListB.shape[0])//4]]
            fNormTrainB = np.concatenate((fNormAB,fNormBB),axis=0)
            fNormTrainB = fNormTrainB.reshape(fNormTrainB.shape[0]*fNormTrainB.shape[1],fNormTrainB.shape[2])
            fNormTargetB = np.concatenate((fNormATargetB,fNormBTargetB),axis=0)
            fNormTargetB = fNormTargetB.reshape(fNormTargetB.shape[0]*fNormTargetB.shape[1],fNormTargetB.shape[2])
            
            fNormA = fNorm.reshape(-1,seqL,fNorm.shape[1])
            fNormATarget = np.roll(fNorm,-rollAmt,axis=0).reshape(-1,seqL,fNorm.shape[1])
            fNormB = np.roll(fNorm,-seqL//2,axis=0).reshape(-1,seqL,fNorm.shape[1])
            fNormBTarget = np.roll(fNorm,(-seqL//2)-rollAmt,axis=0).reshape(-1,seqL,fNorm.shape[1])
            fNormA = fNormA[shufList[(subInd*shufList.shape[0])//4:((subInd+1)*shufList.shape[0])//4]]
            fNormATarget = fNormATarget[shufList[(subInd*shufList.shape[0])//4:((subInd+1)*shufList.shape[0])//4]]
            fNormB = fNormB[shufList[(subInd*shufList.shape[0])//4:((subInd+1)*shufList.shape[0])//4]]
            fNormBTarget = fNormBTarget[shufList[(subInd*shufList.shape[0])//4:((subInd+1)*shufList.shape[0])//4]]
            fNormTrain = np.concatenate((fNormA,fNormB),axis=0)
            fNormTrain = fNormTrain.reshape(fNormTrain.shape[0]*fNormTrain.shape[1],fNormTrain.shape[2])
            fNormTarget = np.concatenate((fNormATarget,fNormBTarget),axis=0)
            fNormTarget = fNormTarget.reshape(fNormTarget.shape[0]*fNormTarget.shape[1],fNormTarget.shape[2])
            
            fNormA = fRookTwo.reshape(-1,seqL,fRookTwo.shape[1])
            fNormATarget = np.roll(fRookTwo,-rollAmt,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
            fNormB = np.roll(fRookTwo,-seqL//2,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
            fNormBTarget = np.roll(fRookTwo,(-seqL//2)-rollAmt,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
            fNormAB = fNormA[shufListC[(subInd*shufListC.shape[0])//4:((subInd+1)*shufListC.shape[0])//4]]
            fNormATargetB = fNormATarget[shufListC[(subInd*shufListC.shape[0])//4:((subInd+1)*shufListC.shape[0])//4]]
            fNormBB = fNormB[shufListC[(subInd*shufListC.shape[0])//4:((subInd+1)*shufListC.shape[0])//4]]
            fNormBTargetB = fNormBTarget[shufListC[(subInd*shufListC.shape[0])//4:((subInd+1)*shufListC.shape[0])//4]]
            fRookTrainB = np.concatenate((fNormAB,fNormBB),axis=0)
            fRookTrainB = fRookTrainB.reshape(fRookTrainB.shape[0]*fRookTrainB.shape[1],fRookTrainB.shape[2])
            fRookTargetB = np.concatenate((fNormATargetB,fNormBTargetB),axis=0)
            fRookTargetB = fRookTargetB.reshape(fRookTargetB.shape[0]*fRookTargetB.shape[1],fRookTargetB.shape[2])
            
            fNormA = fRook.reshape(-1,seqL,fRook.shape[1])
            fNormATarget = np.roll(fRook,-rollAmt,axis=0).reshape(-1,seqL,fRook.shape[1])
            fNormB = np.roll(fRook,-seqL//2,axis=0).reshape(-1,seqL,fRook.shape[1])
            fNormBTarget = np.roll(fRook,(-seqL//2)-rollAmt,axis=0).reshape(-1,seqL,fRook.shape[1])
            fNormA = fNormA[shufListR[(subInd*shufListR.shape[0])//4:((subInd+1)*shufListR.shape[0])//4]]
            fNormATarget = fNormATarget[shufListR[(subInd*shufListR.shape[0])//4:((subInd+1)*shufListR.shape[0])//4]]
            fNormB = fNormB[shufListR[(subInd*shufListR.shape[0])//4:((subInd+1)*shufListR.shape[0])//4]]
            fNormBTarget = fNormBTarget[shufListR[(subInd*shufListR.shape[0])//4:((subInd+1)*shufListR.shape[0])//4]]
            fRookTrain = np.concatenate((fNormA,fNormB),axis=0)
            fRookTrain = fRookTrain.reshape(fRookTrain.shape[0]*fRookTrain.shape[1],fRookTrain.shape[2])
            fRookTarget = np.concatenate((fNormATarget,fNormBTarget),axis=0)
            fRookTarget = fRookTarget.reshape(fRookTarget.shape[0]*fRookTarget.shape[1],fRookTarget.shape[2])
            
            fTotalTrain = np.concatenate((fNormTrain,fRookTrain),axis=0)
            fTotalTarget = np.concatenate((fNormTarget,fRookTarget),axis=0)
            fTotalTrainAlt = np.concatenate((fNormTrain,fNormTrainB),axis=0)
            fTotalTargetAlt = np.concatenate((fNormTarget,fNormTargetB),axis=0)
            fTotalTrainRook = np.concatenate((fRookTrainB,fRookTrain),axis=0)
            fTotalTargetRook = np.concatenate((fRookTargetB,fRookTarget),axis=0)
            # fTotalTrain3 = np.concatenate((fNormTrainB3,fRookTrainB3),axis=0)
            # fTotalTarget3 = np.concatenate((fNormTargetB3,fRookTargetB3),axis=0)
            # fTotalTrain4 = np.concatenate((fNormTrainB4,fRookTrainB4),axis=0)
            # fTotalTarget4 = np.concatenate((fNormTargetB4,fRookTargetB4),axis=0)
            
            # del fNormTrain, fNormTarget, fRookTrain, fRookTarget, fRookTrainB, fRookTargetB, fNormA, fNormATarget, fNormB, fNormBTarget
            # del fNormAB, fNormATargetB, fNormBB, fNormBTargetB, fNormTrainB, fNormTargetB, fNormTrainB3, fNormTargetB3, fRookTrainB3, fRookTargetB3, fRookTrainB4, fRookTargetB4
            del fNormTrain, fNormTarget, fRookTrain, fRookTarget, fRookTrainB, fRookTargetB, fNormA, fNormATarget, fNormB, fNormBTarget
            del fNormAB, fNormATargetB, fNormBB, fNormBTargetB, fNormTrainB, fNormTargetB
            ##print 'total features train shape: ' + str(fTotalTrain.shape)
            ##print 'total features target shape: ' + str(fTotalTarget.shape)
            ##print 'total featuresAlt train shape: ' + str(fTotalTrain.shape)
            ##print 'total featuresAlt target shape: ' + str(fTotalTarget.shape)
            ##print 'total featuresRook train shape: ' + str(fTotalTrainRook.shape)
            ##print 'total featuresRook target shape: ' + str(fTotalTargetRook.shape)
            ##print 'total features 3 train shape: ' + str(fTotalTrain3.shape)
            ##print 'total features 3 target shape: ' + str(fTotalTarget3.shape)
            ##print 'total features 4 train shape: ' + str(fTotalTrain4.shape)
            ##print 'total features 4 target shape: ' + str(fTotalTarget4.shape)
            #filterSize = 32
            nnTrack = []
            rnTrack = []
            rrTrack = []
            inputs = Input((seqL,fTotalTrain.shape[1],))
            x = Conv1D(filterSize,4,dilation_rate=1,padding='causal',activation='tanh')(inputs)
            x = Conv1D(filterSize,4,dilation_rate=2,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=4,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=8,padding='causal',activation='tanh')(x)
            x = TimeDistributed(Dense(fTotalTrain.shape[1],activation='tanh'))(x)
            model = Model(inputs=inputs,outputs=x)
            model.compile(loss='mean_squared_error', optimizer='adam')
            ##print '- Model Init Cell Complete -'
            for i in range(trainIters):
                rI = np.random.randint(fTotalTrain.shape[0]//seqL)*seqL
                xTrain = np.roll(fTotalTrain,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
                yTrain = np.roll(fTotalTarget,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
                history = model.fit(xTrain,yTrain,batch_size=batchSize,shuffle=False,epochs=10)
                rnTrack = np.concatenate((rnTrack,history.history['loss']))
            model.save('models/' + modelVersion + '-' + fileSuffix + '/23modelCombo' + str(topInd) + '-' + str(subInd) + '.h5')
            del model
            ##print '- Training Cell Complete -'
            inputs = Input((seqL,fTotalTrainAlt.shape[1],))
            x = Conv1D(filterSize,4,dilation_rate=1,padding='causal',activation='tanh')(inputs)
            x = Conv1D(filterSize,4,dilation_rate=2,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=4,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=8,padding='causal',activation='tanh')(x)
            x = TimeDistributed(Dense(fTotalTrainAlt.shape[1],activation='tanh'))(x)
            model = Model(inputs=inputs,outputs=x)
            model.compile(loss='mean_squared_error', optimizer='adam')
            ##print '- Model 2 Init Cell Complete -'
            for i in range(trainIters):
                rI = np.random.randint(fTotalTrainAlt.shape[0]//seqL)*seqL
                xTrain = np.roll(fTotalTrainAlt,rI,axis=0)[:seqL*(fTotalTrainAlt.shape[0]//seqL)].reshape(-1,seqL,fTotalTrainAlt.shape[1])
                yTrain = np.roll(fTotalTargetAlt,rI,axis=0)[:seqL*(fTotalTrainAlt.shape[0]//seqL)].reshape(-1,seqL,fTotalTrainAlt.shape[1])
                history = model.fit(xTrain,yTrain,batch_size=batchSize,shuffle=False,epochs=10)
                nnTrack = np.concatenate((nnTrack,history.history['loss']))
            model.save('models/' + modelVersion + '-' + fileSuffix + '/23modelNN' + str(topInd) + '-' + str(subInd) + '.h5')
            del model
            ##print '- Training 2 Cell Complete -'
            inputs = Input((seqL,fTotalTrainRook.shape[1],))
            x = Conv1D(filterSize,4,dilation_rate=1,padding='causal',activation='tanh')(inputs)
            x = Conv1D(filterSize,4,dilation_rate=2,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=4,padding='causal',activation='tanh')(x)
            x = Conv1D(filterSize,4,dilation_rate=8,padding='causal',activation='tanh')(x)
            x = TimeDistributed(Dense(fTotalTrainRook.shape[1],activation='tanh'))(x)
            model = Model(inputs=inputs,outputs=x)
            model.compile(loss='mean_squared_error', optimizer='adam')
            ##print '- Model 3 Init Cell Complete -'
            for i in range(trainIters):
                rI = np.random.randint(fTotalTrainRook.shape[0]//seqL)*seqL
                xTrain = np.roll(fTotalTrainRook,rI,axis=0)[:seqL*(fTotalTrainRook.shape[0]//seqL)].reshape(-1,seqL,fTotalTrainRook.shape[1])
                yTrain = np.roll(fTotalTargetRook,rI,axis=0)[:seqL*(fTotalTrainRook.shape[0]//seqL)].reshape(-1,seqL,fTotalTrainRook.shape[1])
                history = model.fit(xTrain,yTrain,batch_size=batchSize,shuffle=False,epochs=10)
                rrTrack = np.concatenate((rrTrack,history.history['loss']))
            model.save('models/' + modelVersion + '-' + fileSuffix + '/23modelRR' + str(topInd) + '-' + str(subInd) + '.h5')
            del model
            ##print '- Training 3 Cell Complete -'
            model = load_model('models/' + modelVersion + '-' + fileSuffix + '/23modelCombo' + str(topInd) + '-' + str(subInd) + '.h5')
            modelB = load_model('models/' + modelVersion + '-' + fileSuffix + '/23modelNN' + str(topInd) + '-' + str(subInd) + '.h5')
            modelC = load_model('models/' + modelVersion + '-' + fileSuffix + '/23modelRR' + str(topInd) + '-' + str(subInd) + '.h5')
            l2NormList = []
            l2RookList = []
            l2NormAltList = []
            l2NormBList = []
            l2RookAltList = []
            l2RookBList = []
            l2Norm3OnNN = []
            l2Norm4OnNN = []
            l2Rook3OnNN = []
            l2Rook4OnNN = []
            l2Norm3OnRR = []
            l2Norm4OnRR = []
            l2Rook3OnRR = []
            l2Rook4OnRR = []
            for j in range(1):

                fNormA = fNormTwo.reshape(-1,seqL,fNormTwo.shape[1])
                fNormATarget = np.roll(fNormTwo,-1,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormB = np.roll(fNormTwo,-seqL//2,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormBTarget = np.roll(fNormTwo,(-seqL//2)-1,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormAB = fNormA[shufListB[(((subInd+2)%4)*shufListB.shape[0])//4:((((subInd+2)%4)+1)*shufListB.shape[0])//4]]
                fNormATargetB = fNormATarget[shufListB[(((subInd+2)%4)*shufListB.shape[0])//4:((((subInd+2)%4)+1)*shufListB.shape[0])//4]]
                fNormBB = fNormB[shufListB[(((subInd+2)%4)*shufListB.shape[0])//4:((((subInd+2)%4)+1)*shufListB.shape[0])//4]]
                fNormBTargetB = fNormBTarget[shufListB[(((subInd+2)%4)*shufListB.shape[0])//4:((((subInd+2)%4)+1)*shufListB.shape[0])//4]]
                fNormTrainB = np.concatenate((fNormAB,fNormBB),axis=0)
                fNormTrainB = fNormTrainB.reshape(fNormTrainB.shape[0]*fNormTrainB.shape[1],fNormTrainB.shape[2])
                fNormTargetB = np.concatenate((fNormATargetB,fNormBTargetB),axis=0)
                fNormTargetB = fNormTargetB.reshape(fNormTargetB.shape[0]*fNormTargetB.shape[1],fNormTargetB.shape[2])
                newTotalTrain3 = fNormTrainB
                newTotalTrain3 = newTotalTrain3[:(newTotalTrain3.shape[0]//seqL)*seqL,:]
                newTotalTrain3 = newTotalTrain3.reshape(-1,seqL,newTotalTrain3.shape[1])
                newTotalTarget3 = fNormTargetB
                newTotalTarget3 = newTotalTarget3[:(newTotalTarget3.shape[0]//seqL)*seqL,:]
                newTotalTarget3 = newTotalTarget3.reshape(-1,seqL,newTotalTarget3.shape[1])
                trainList3 = np.arange(newTotalTrain3.shape[0])
                np.random.shuffle(trainList3)
                fNormVal3 = newTotalTrain3[trainList3,:,:]
                fNormTar3 = newTotalTarget3[trainList3,:,:]
                fNormVal3 = fNormVal3.reshape(fNormVal3.shape[0]*fNormVal3.shape[1],fNormVal3.shape[2])
                fNormTar3 = fNormTar3.reshape(fNormTar3.shape[0]*fNormTar3.shape[1],fNormTar3.shape[2])

                fNormA = fNormTwo.reshape(-1,seqL,fNormTwo.shape[1])
                fNormATarget = np.roll(fNormTwo,-1,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormB = np.roll(fNormTwo,-seqL//2,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormBTarget = np.roll(fNormTwo,(-seqL//2)-1,axis=0).reshape(-1,seqL,fNormTwo.shape[1])
                fNormAB = fNormA[shufListB[(((subInd+3)%4)*shufListB.shape[0])//4:((((subInd+3)%4)+1)*shufListB.shape[0])//4]]
                fNormATargetB = fNormATarget[shufListB[(((subInd+3)%4)*shufListB.shape[0])//4:((((subInd+3)%4)+1)*shufListB.shape[0])//4]]
                fNormBB = fNormB[shufListB[(((subInd+3)%4)*shufListB.shape[0])//4:((((subInd+3)%4)+1)*shufListB.shape[0])//4]]
                fNormBTargetB = fNormBTarget[shufListB[(((subInd+3)%4)*shufListB.shape[0])//4:((((subInd+3)%4)+1)*shufListB.shape[0])//4]]
                fNormTrainB = np.concatenate((fNormAB,fNormBB),axis=0)
                fNormTrainB = fNormTrainB.reshape(fNormTrainB.shape[0]*fNormTrainB.shape[1],fNormTrainB.shape[2])
                fNormTargetB = np.concatenate((fNormATargetB,fNormBTargetB),axis=0)
                fNormTargetB = fNormTargetB.reshape(fNormTargetB.shape[0]*fNormTargetB.shape[1],fNormTargetB.shape[2])
                newTotalTrain3 = fNormTrainB
                newTotalTrain3 = newTotalTrain3[:(newTotalTrain3.shape[0]//seqL)*seqL,:]
                newTotalTrain3 = newTotalTrain3.reshape(-1,seqL,newTotalTrain3.shape[1])
                newTotalTarget3 = fNormTargetB
                newTotalTarget3 = newTotalTarget3[:(newTotalTarget3.shape[0]//seqL)*seqL,:]
                newTotalTarget3 = newTotalTarget3.reshape(-1,seqL,newTotalTarget3.shape[1])
                trainList3 = np.arange(newTotalTrain3.shape[0])
                np.random.shuffle(trainList3)
                fNormVal4 = newTotalTrain3[trainList3,:,:]
                fNormTar4 = newTotalTarget3[trainList3,:,:]
                fNormVal4 = fNormVal4.reshape(fNormVal4.shape[0]*fNormVal4.shape[1],fNormVal4.shape[2])
                fNormTar4 = fNormTar4.reshape(fNormTar4.shape[0]*fNormTar4.shape[1],fNormTar4.shape[2])
                
                fNormA = fRookTwo.reshape(-1,seqL,fRookTwo.shape[1])
                fNormATarget = np.roll(fRookTwo,-1,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
                fNormB = np.roll(fRookTwo,-seqL//2,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
                fNormBTarget = np.roll(fRookTwo,(-seqL//2)-1,axis=0).reshape(-1,seqL,fRookTwo.shape[1])
                fNormAB = fNormA[shufListC[(((subInd+2)%4)*shufListC.shape[0])//4:((((subInd+2)%4)+1)*shufListC.shape[0])//4]]
                fNormATargetB = fNormATarget[shufListC[(((subInd+2)%4)*shufListC.shape[0])//4:((((subInd+2)%4)+1)*shufListC.shape[0])//4]]
                fNormBB = fNormB[shufListC[(((subInd+2)%4)*shufListC.shape[0])//4:((((subInd+2)%4)+1)*shufListC.shape[0])//4]]
                fNormBTargetB = fNormBTarget[shufListC[(((subInd+2)%4)*shufListC.shape[0])//4:((((subInd+2)%4)+1)*shufListC.shape[0])//4]]
                fRookTrainB = np.concatenate((fNormAB,fNormBB),axis=0)
                fRookTrainB = fRookTrainB.reshape(fRookTrainB.shape[0]*fRookTrainB.shape[1],fRookTrainB.shape[2])
                fRookTargetB = np.concatenate((fNormATargetB,fNormBTargetB),axis=0)
                fRookTargetB = fRookTargetB.reshape(fRookTargetB.shape[0]*fRookTargetB.shape[1],fRookTargetB.shape[2])
                rookTotalTrain3 = fRookTrainB
                rookTotalTrain3 = rookTotalTrain3[:(rookTotalTrain3.shape[0]//seqL)*seqL,:]
                rookTotalTrain3 = rookTotalTrain3.reshape(-1,seqL,rookTotalTrain3.shape[1])
                rookTotalTarget3 = fRookTargetB
                rookTotalTarget3 = rookTotalTarget3[:(rookTotalTarget3.shape[0]//seqL)*seqL,:]
                rookTotalTarget3 = rookTotalTarget3.reshape(-1,seqL,rookTotalTarget3.shape[1])
                rookTrainList3 = np.arange(rookTotalTrain3.shape[0])
                np.random.shuffle(rookTrainList3)
                fRookVal3 = rookTotalTrain3[rookTrainList3,:,:]
                fRookTar3 = rookTotalTarget3[rookTrainList3,:,:]
                fRookVal3 = fRookVal3.reshape(fRookVal3.shape[0]*fRookVal3.shape[1],fRookVal3.shape[2])
                fRookTar3 = fRookTar3.reshape(fRookTar3.shape[0]*fRookTar3.shape[1],fRookTar3.shape[2])
                
                newTotalTrain = fTotalTrain[:fTotalTrain.shape[0]//2]
                newTotalTrain = newTotalTrain[:(newTotalTrain.shape[0]//seqL)*seqL,:]
                newTotalTrain = newTotalTrain.reshape(-1,seqL,newTotalTrain.shape[1])
                newTotalTarget = fTotalTarget[:fTotalTarget.shape[0]//2]
                newTotalTarget = newTotalTarget[:(newTotalTarget.shape[0]//seqL)*seqL,:]
                newTotalTarget = newTotalTarget.reshape(-1,seqL,newTotalTarget.shape[1])
                trainList = np.arange(newTotalTrain.shape[0])
                np.random.shuffle(trainList)
                fNormVal = newTotalTrain[trainList,:,:]
                fNormTar = newTotalTarget[trainList,:,:]
                fNormVal = fNormVal.reshape(fNormVal.shape[0]*fNormVal.shape[1],fNormVal.shape[2])
                fNormTar = fNormTar.reshape(fNormTar.shape[0]*fNormTar.shape[1],fNormTar.shape[2])
                
                bTotalTrain = fTotalTrainAlt[fTotalTrainAlt.shape[0]//2:]
                bTotalTrain = bTotalTrain[:(bTotalTrain.shape[0]//seqL)*seqL,:]
                bTotalTrain = bTotalTrain.reshape(-1,seqL,bTotalTrain.shape[1])
                bTotalTarget = fTotalTargetAlt[fTotalTargetAlt.shape[0]//2:]
                bTotalTarget = bTotalTarget[:(bTotalTarget.shape[0]//seqL)*seqL,:]
                bTotalTarget = bTotalTarget.reshape(-1,seqL,bTotalTarget.shape[1])
                bTrainList = np.arange(bTotalTrain.shape[0])
                np.random.shuffle(bTrainList)
                fbVal = bTotalTrain[bTrainList,:,:]
                fbTar = bTotalTarget[bTrainList,:,:]
                fbVal = fbVal.reshape(fbVal.shape[0]*fbVal.shape[1],fbVal.shape[2])
                fbTar = fbTar.reshape(fbTar.shape[0]*fbTar.shape[1],fbTar.shape[2])
                
                rookTotalTrain = fTotalTrain[fTotalTrain.shape[0]//2:]
                rookTotalTrain = rookTotalTrain[:(rookTotalTrain.shape[0]//seqL)*seqL,:]
                rookTotalTrain = rookTotalTrain.reshape(-1,seqL,rookTotalTrain.shape[1])
                rookTotalTarget = fTotalTarget[fTotalTarget.shape[0]//2:]
                rookTotalTarget = rookTotalTarget[:(rookTotalTarget.shape[0]//seqL)*seqL,:]
                rookTotalTarget = rookTotalTarget.reshape(-1,seqL,rookTotalTarget.shape[1])
                rookTrainList = np.arange(rookTotalTrain.shape[0])
                np.random.shuffle(rookTrainList)
                fRookVal = rookTotalTrain[rookTrainList,:,:]
                fRookTar = rookTotalTarget[rookTrainList,:,:]
                fRookVal = fRookVal.reshape(fRookVal.shape[0]*fRookVal.shape[1],fRookVal.shape[2])
                fRookTar = fRookTar.reshape(fRookTar.shape[0]*fRookTar.shape[1],fRookTar.shape[2])
                
                rookTotalTrainB = fTotalTrainRook[:fTotalTrainRook.shape[0]//2]
                rookTotalTrainB = rookTotalTrainB[:(rookTotalTrainB.shape[0]//seqL)*seqL,:]
                rookTotalTrainB = rookTotalTrainB.reshape(-1,seqL,rookTotalTrainB.shape[1])
                rookTotalTargetB = fTotalTargetRook[:fTotalTargetRook.shape[0]//2]
                rookTotalTargetB = rookTotalTargetB[:(rookTotalTargetB.shape[0]//seqL)*seqL,:]
                rookTotalTargetB = rookTotalTargetB.reshape(-1,seqL,rookTotalTargetB.shape[1])
                rookTrainListB = np.arange(rookTotalTrainB.shape[0])
                np.random.shuffle(rookTrainListB)
                fRookValB = rookTotalTrainB[rookTrainListB,:,:]
                fRookTarB = rookTotalTargetB[rookTrainListB,:,:]
                fRookValB = fRookValB.reshape(fRookValB.shape[0]*fRookValB.shape[1],fRookValB.shape[2])
                fRookTarB = fRookTarB.reshape(fRookTarB.shape[0]*fRookTarB.shape[1],fRookTarB.shape[2])
                
                sizeList = []
                sizeList.append(fNormVal.shape[0])
                sizeList.append(fRookVal.shape[0])
                sizeList.append(fNormVal3.shape[0])
                sizeList.append(fNormVal4.shape[0])
                sizeList.append(fRookVal3.shape[0])
                sizeList.append(fbVal.shape[0])
                sizeList.append(fRookValB.shape[0])
                sizeMin = min(sizeList)
                fNormVal = fNormVal[:sizeMin]
                fNormTar = fNormTar[:sizeMin]
                fRookVal = fRookVal[:sizeMin]
                fRookTar = fRookTar[:sizeMin]
                fNormVal3 = fNormVal3[:sizeMin]
                fNormTar3 = fNormTar3[:sizeMin]
                fNormVal4 = fNormVal4[:sizeMin]
                fNormTar4 = fNormTar4[:sizeMin]
                fRookVal3 = fRookVal3[:sizeMin]
                fRookTar3 = fRookTar3[:sizeMin]
                fbVal = fbVal[:sizeMin]
                fbTar = fbTar[:sizeMin]
                fRookValB = fRookValB[:sizeMin]
                fRookTarB = fRookTarB[:sizeMin]

                fNormVal = fNormVal.reshape(-1,seqL,fNormVal.shape[1])
                fRookVal = fRookVal.reshape(-1,seqL,fRookVal.shape[1])
                fNormVal3 = fNormVal3.reshape(-1,seqL,fNormVal3.shape[1])
                fNormVal4 = fNormVal3.reshape(-1,seqL,fNormVal4.shape[1])
                fRookVal3 = fRookVal3.reshape(-1,seqL,fRookVal3.shape[1])
                fbVal = fbVal.reshape(-1,seqL,fbVal.shape[1])
                fRookValB = fRookValB.reshape(-1,seqL,fRookValB.shape[1])
                
                fNormTar = fNormTar.reshape(-1,seqL,fNormTar.shape[1])
                fRookTar = fRookTar.reshape(-1,seqL,fRookTar.shape[1])
                fNormTar3 = fNormTar3.reshape(-1,seqL,fNormTar3.shape[1])
                fNormTar4 = fNormTar3.reshape(-1,seqL,fNormTar4.shape[1])
                fRookTar3 = fRookTar3.reshape(-1,seqL,fRookTar3.shape[1])
                fbTar = fbTar.reshape(-1,seqL,fbTar.shape[1])
                fRookTarB = fRookTarB.reshape(-1,seqL,fRookTarB.shape[1])

            
                normPred = model.predict(fNormVal)
                l2NormList.append(np.sqrt(np.sum((normPred[:,(-seqL//2):,:] - fNormTar[:,-(seqL//2):,:])**2,axis=-1)))
                rookPred = model.predict(fRookVal)
                l2RookList.append(np.sqrt(np.sum((rookPred[:,(-seqL//2):,:] - fRookTar[:,-(seqL//2):,:])**2,axis=-1)))
                
                normPred3 = modelB.predict(fNormVal3)
                l2Norm3OnNN.append(np.sqrt(np.sum((normPred3[:,(-seqL//2):,:] - fNormTar3[:,-(seqL//2):,:])**2,axis=-1)))
                rookPred3 = modelB.predict(fRookVal3)
                l2Rook3OnNN.append(np.sqrt(np.sum((rookPred3[:,(-seqL//2):,:] - fRookTar3[:,-(seqL//2):,:])**2,axis=-1)))
                normPred4 = modelB.predict(fNormVal4)
                l2Norm4OnNN.append(np.sqrt(np.sum((normPred4[:,(-seqL//2):,:] - fNormTar4[:,-(seqL//2):,:])**2,axis=1)))

                normPred3 = modelC.predict(fNormVal3)
                l2Norm3OnRR.append(np.sqrt(np.sum((normPred3[:,(-seqL//2):,:] - fNormTar3[:,-(seqL//2):,:])**2,axis=-1)))
                rookPred3 = modelC.predict(fRookVal3)
                l2Rook3OnRR.append(np.sqrt(np.sum((rookPred3[:,(-seqL//2):,:] - fRookTar3[:,-(seqL//2):,:])**2,axis=-1)))
                normPred4 = modelC.predict(fNormVal4)
                l2Norm4OnRR.append(np.sqrt(np.sum((normPred4[:,(-seqL//2):,:] - fNormTar4[:,-(seqL//2):,:])**2,axis=1)))
                                
                normAltPred = modelB.predict(fNormVal)
                l2NormAltList.append(np.sqrt(np.sum((normAltPred[:,(-seqL//2):,:] - fNormTar[:,-(seqL//2):,:])**2,axis=-1)))
                normBPred = modelB.predict(fbVal)
                l2NormBList.append(np.sqrt(np.sum((normBPred[:,(-seqL//2):,:] - fbTar[:,-(seqL//2):,:])**2,axis=-1)))
                
                rookAltPred = modelC.predict(fRookVal)
                l2RookAltList.append(np.sqrt(np.sum((rookAltPred[:,(-seqL//2):,:] - fRookTar[:,-(seqL//2):,:])**2,axis=-1)))
                rookBPred = modelC.predict(fRookValB)
                l2RookBList.append(np.sqrt(np.sum((rookBPred[:,(-seqL//2):,:] - fRookTarB[:,-(seqL//2):,:])**2,axis=-1)))

            normBNormComp = []
            rookNormComp = []
            rookBRookComp = []
            # l2Norm4OnNN = l2Norm3OnNN
            l2Rook4OnNN = l2Rook3OnNN
            # l2Norm4OnRR = l2Norm3OnRR
            l2Rook4OnRR = l2Rook3OnRR

            l2NormList      = np.hstack(l2NormList)
            l2RookList      = np.hstack(l2RookList)
            l2Norm3OnNN     = np.hstack(l2Norm3OnNN)
            l2Norm4OnNN     = np.hstack(l2Norm4OnNN)
            l2Rook3OnNN     = np.hstack(l2Rook3OnNN)
            l2Norm3OnRR     = np.hstack(l2Norm3OnRR)
            l2Norm4OnRR     = np.hstack(l2Norm4OnRR)
            l2Rook3OnRR     = np.hstack(l2Rook3OnRR)
            l2NormAltList   = np.hstack(l2NormAltList)
            l2RookAltList   = np.hstack(l2RookAltList)
            l2NormBList     = np.hstack(l2NormBList)
            l2RookBList     = np.hstack(l2RookBList)

            l2NormListHolder.append(l2NormList)
            l2RookListHolder.append(l2RookList)
            l2Norm3OnNNHolder.append(l2Norm3OnNN)
            l2Norm4OnNNHolder.append(l2Norm4OnNN)
            l2Rook3OnNNHolder.append(l2Rook3OnNN)
            l2Norm3OnRRHolder.append(l2Norm3OnRR)
            l2Norm4OnRRHolder.append(l2Norm4OnRR)
            l2Rook3OnRRHolder.append(l2Rook3OnRR)
            l2NormBListHolder.append(l2NormBList)
            l2RookBListHolder.append(l2RookBList)
            l2NormAltListHolder.append(l2NormAltList)
            l2RookAltListHolder.append(l2RookAltList)

            

            ##print 'NormB Mean: ' + str(np.mean(np.array(l2NormBList)))
            ##print 'NormAlt Mean: ' + str(np.mean(np.array(l2NormAltList)))
            ##print 'RookB Mean: ' + str(np.mean(np.array(l2RookBList)))
            ##print 'RookAlt Mean: ' + str(np.mean(np.array(l2RookAltList)))
            ##print 'Rook Mean: ' + str(np.mean(np.array(l2RookList)))
            ##print 'Norm Mean: ' + str(np.mean(np.array(l2NormList)))
            meanAvgRookvNorm = (np.mean(np.array(l2NormList)) + np.mean(np.array(l2RookList))) / 2.0
            meanAvgNormvNorm = (np.mean(np.array(l2NormAltList)) + np.mean(np.array(l2NormBList))) / 2.0
            meanAvgRookvRook = (np.mean(np.array(l2RookAltList)) + np.mean(np.array(l2RookBList))) / 2.0
            ##print 'Avg Mean (NormB v NormAlt): ' + str(meanAvgNormvNorm)
            ##print 'Avg Mean (RookB v RookAlt): ' + str(meanAvgRookvRook)
            ##print 'Avg Mean (Rook v Norm): ' + str(meanAvgRookvNorm)
            ##print 'NormB StdDev: ' + str(np.std(np.array(l2NormBList)))
            #print 'NormAlt StdDev: ' + str(np.std(np.array(l2NormAltList)))
            #print 'RookB StdDev: ' + str(np.std(np.array(l2RookBList)))
            #print 'RookAlt StdDev: ' + str(np.std(np.array(l2RookAltList)))
            #print 'Rook StdDev: ' + str(np.std(np.array(l2RookList)))
            #print 'Norm StdDev: ' + str(np.std(np.array(l2NormList)))
            #print 'NormB RMS: ' + str(np.sqrt(np.mean(np.array(l2NormBList)**2)))
            #print 'NormAlt RMS: ' + str(np.sqrt(np.mean(np.array(l2NormAltList)**2)))
            #print 'RookB RMS: ' + str(np.sqrt(np.mean(np.array(l2RookBList)**2)))
            #print 'RookAlt RMS: ' + str(np.sqrt(np.mean(np.array(l2RookAltList)**2)))
            #print 'Rook RMS: ' + str(np.sqrt(np.mean(np.array(l2RookList)**2)))
            #print 'Norm RMS: ' + str(np.sqrt(np.mean(np.array(l2NormList)**2)))
            totalRMSRookvNorm = np.sqrt(np.mean(np.concatenate((np.array(l2RookList)**2,np.array(l2NormList)**2))))
            totalRMSNormvNorm = np.sqrt(np.mean(np.concatenate((np.array(l2NormBList)**2,np.array(l2NormAltList)**2))))
            totalRMSRookvRook = np.sqrt(np.mean(np.concatenate((np.array(l2RookBList)**2,np.array(l2RookAltList)**2))))
            #print 'Total RMS (NormB v NormAlt): ' + str(totalRMSNormvNorm)
            #print 'Total RMS (RookB v RookAlt): ' + str(totalRMSRookvRook)
            #print 'Total RMS (Rook v Norm): ' + str(totalRMSRookvNorm)
            #print 'NormB vs NormAlt Scaled Mean Comparison (should be low, around 0): ' + str((np.mean(np.array(l2NormBList)) - np.mean(np.array(l2NormAltList))) / meanAvgNormvNorm)
            #print 'RookB vs RookAlt Scaled Mean Comparison (should be low, around 0): ' + str((np.mean(np.array(l2RookBList)) - np.mean(np.array(l2RookAltList))) / meanAvgRookvRook)
            #print 'Rook vs Norm Scaled Mean Comparison (should be higher): ' + str((np.mean(np.array(l2RookList)) - np.mean(np.array(l2NormList))) / meanAvgRookvNorm)
            #print 'NormB vs NormAlt Scaled RMS Comparison (should be low, around 0): ' + str((np.sqrt(np.mean(np.array(l2NormBList)**2)) - np.sqrt(np.mean(np.array(l2NormAltList)**2))) / totalRMSNormvNorm)
            #print 'RookB vs RookAlt Scaled RMS Comparison (should be low, around 0): ' + str((np.sqrt(np.mean(np.array(l2RookBList)**2)) - np.sqrt(np.mean(np.array(l2RookAltList)**2))) / totalRMSRookvRook)
            #print 'Rook vs Norm Scaled RMS Comparison (should be higher): ' + str((np.sqrt(np.mean(np.array(l2RookList)**2)) - np.sqrt(np.mean(np.array(l2NormList)**2))) / totalRMSRookvNorm)
            #print 'NormB vs NormAlt RMS of Comparisons (should be low, around 0): ' + str(np.sqrt(np.abs(np.mean(np.array(l2NormBList)**2) - np.mean(np.array(l2NormAltList)**2))) / totalRMSNormvNorm)
            #print 'RookB vs RookAlt RMS of Comparisons (should be low, around 0): ' + str(np.sqrt(np.abs(np.mean(np.array(l2RookBList)**2) - np.mean(np.array(l2RookAltList)**2))) / totalRMSRookvRook)
            #print 'Rook vs Norm RMS of Comparisons (should be higher): ' + str(np.sqrt(np.abs(np.mean(np.array(l2RookList)**2) - np.mean(np.array(l2NormList)**2))) / totalRMSRookvNorm)
            #print 'NormB vs NormAlt Binary Comparisons (should be low, around 0): ' + str(float(((np.array(l2NormBList) - np.array(l2NormAltList)) >0).sum()) / float((np.array(l2NormBList) - np.array(l2NormAltList)).shape[0]))
            #print 'RookB vs RookAlt Binary Comparisons (should be low, around 0): ' + str(float(((np.array(l2RookBList) - np.array(l2RookAltList)) >0).sum()) / float((np.array(l2RookBList) - np.array(l2RookAltList)).shape[0]))
            #print 'Rook vs Norm Binary Comparisons (should be higher): ' + str(float(((np.array(l2RookList) - np.array(l2NormList)) >0).sum()) / float((np.array(l2RookList) - np.array(l2NormList)).shape[0]))
            #print 'NormB vs Norm Loss RMS: ' + str(np.sqrt(np.mean(np.array(nnTrack)**2)))
            #print 'RookB vs Rook Loss RMS: ' + str(np.sqrt(np.mean(np.array(rrTrack)**2)))
            #print 'Rook vs Norm Loss RMS: ' + str(np.sqrt(np.mean(np.array(rnTrack)**2)))
            for i in range(int(len(l2NormList)-((len(l2NormList)//4)+1))):
                normBNormComp.append((np.mean(np.array(l2NormBList[i:i+len(l2NormBList)//4])) - np.mean(np.array(l2NormAltList[i:i+len(l2NormAltList)//4]))) / meanAvgNormvNorm)
                rookBRookComp.append((np.mean(np.array(l2RookBList[i:i+len(l2RookBList)//4])) - np.mean(np.array(l2RookAltList[i:i+len(l2RookAltList)//4]))) / meanAvgRookvRook)
                rookNormComp.append((np.mean(np.array(l2RookList[i:i+len(l2RookList)//4])) - np.mean(np.array(l2NormList[i:i+len(l2NormList)//4]))) / meanAvgRookvNorm)
            plt.plot(normBNormComp,label='NormB v NormAlt')
            plt.plot(rookBRookComp,label='RookB v RookAlt')
            plt.plot(rookNormComp,label='Rook v Norm')
            plt.legend()
            #plt.savefig('predictionErrorComps' + str(topInd) + '-' + str(subInd) + '.png')
            # plt.show()
            plt.savefig('results/' + resultsFileName + '-PredErrorComps' + str(topInd) + '-' + str(subInd) + '.png')
            plt.clf()
            plt.plot(nnTrack,label='nnLoss')
            plt.plot(rrTrack,label='rrLoss')
            plt.plot(rnTrack,label='rnLoss')
            plt.legend()
            #plt.savefig('modelLossComps' + str(topInd) + '-' + str(subInd) + '.png')
            # plt.show()
            plt.savefig('results/' + resultsFileName + '-LossComps' + str(topInd) + '-' + str(subInd) + '.png')
            plt.clf()
            csv_out = []
            csv_out.append(np.mean(np.array(l2NormBList)))
            csv_out.append(np.mean(np.array(l2NormAltList)))
            csv_out.append(np.mean(np.array(l2RookBList)))
            csv_out.append(np.mean(np.array(l2RookAltList)))
            csv_out.append(np.mean(np.array(l2RookList)))
            csv_out.append(np.mean(np.array(l2NormList)))
            meanAvgRookvNorm = (np.mean(np.array(l2NormList)) + np.mean(np.array(l2RookList))) / 2.0
            meanAvgNormvNorm = (np.mean(np.array(l2NormAltList)) + np.mean(np.array(l2NormBList))) / 2.0
            meanAvgRookvRook = (np.mean(np.array(l2RookAltList)) + np.mean(np.array(l2RookBList))) / 2.0
            csv_out.append(meanAvgNormvNorm)
            csv_out.append(meanAvgRookvRook)
            csv_out.append(meanAvgRookvNorm)
            csv_out.append(np.std(np.array(l2NormBList)))
            csv_out.append(np.std(np.array(l2NormAltList)))
            csv_out.append(np.std(np.array(l2RookBList)))
            csv_out.append(np.std(np.array(l2RookAltList)))
            csv_out.append(np.std(np.array(l2RookList)))
            csv_out.append(np.std(np.array(l2NormList)))
            csv_out.append(np.sqrt(np.mean(np.array(l2NormBList)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2NormAltList)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2RookBList)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2RookAltList)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2RookList)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2NormList)**2)))
            totalRMSRookvNorm = np.sqrt(np.mean(np.concatenate((np.array(l2RookList)**2,np.array(l2NormList)**2))))
            totalRMSNormvNorm = np.sqrt(np.mean(np.concatenate((np.array(l2NormBList)**2,np.array(l2NormAltList)**2))))
            totalRMSRookvRook = np.sqrt(np.mean(np.concatenate((np.array(l2RookBList)**2,np.array(l2RookAltList)**2))))
            csv_out.append(totalRMSNormvNorm)
            csv_out.append(totalRMSRookvRook)
            csv_out.append(totalRMSRookvNorm)
            csv_out.append((np.mean(np.array(l2NormBList)) - np.mean(np.array(l2NormAltList))) / meanAvgNormvNorm)
            csv_out.append((np.mean(np.array(l2RookBList)) - np.mean(np.array(l2RookAltList))) / meanAvgRookvRook)
            csv_out.append((np.mean(np.array(l2RookList)) - np.mean(np.array(l2NormList))) / meanAvgRookvNorm)
            csv_out.append((np.sqrt(np.mean(np.array(l2NormBList)**2)) - np.sqrt(np.mean(np.array(l2NormAltList)**2))) / totalRMSNormvNorm)
            csv_out.append((np.sqrt(np.mean(np.array(l2RookBList)**2)) - np.sqrt(np.mean(np.array(l2RookAltList)**2))) / totalRMSRookvRook)
            csv_out.append((np.sqrt(np.mean(np.array(l2RookList)**2)) - np.sqrt(np.mean(np.array(l2NormList)**2))) / totalRMSRookvNorm)
            csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2NormBList)**2) - np.mean(np.array(l2NormAltList)**2))) * np.sign(np.mean(np.array(l2NormBList)**2) - np.mean(np.array(l2NormAltList)**2)) ) / totalRMSNormvNorm)
            csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2RookBList)**2) - np.mean(np.array(l2RookAltList)**2))) * np.sign(np.mean(np.array(l2RookBList)**2) - np.mean(np.array(l2RookAltList)**2)) ) / totalRMSRookvRook)
            csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2RookList)**2) - np.mean(np.array(l2NormList)**2))) * np.sign(np.mean(np.array(l2RookList)**2) - np.mean(np.array(l2NormList)**2)) ) / totalRMSRookvNorm)
            #csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2NormBList)*np.abs(np.array(l2NormBList)) - np.array(l2NormAltList)*np.abs(np.array(l2NormAltList))))) * np.sign(np.mean(np.array(l2NormBList)*np.abs(np.array(l2NormBList)) - np.array(l2NormAltList)*np.abs(np.array(l2NormAltList))))) / totalRMSNormvNorm)
            #csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2RookBList)*np.abs(np.array(l2RookBList)) - np.array(l2RookAltList)*np.abs(np.array(l2RookAltList))))) * np.sign(np.mean(np.array(l2RookBList)*np.abs(np.array(l2RookBList)) - np.array(l2RookAltList)*np.abs(np.array(l2RookAltList))))) / totalRMSRookvRook)
            #csv_out.append( (np.sqrt(np.abs(np.mean(np.array(l2RookList)*np.abs(np.array(l2RookList)) - np.array(l2NormList)*np.abs(np.array(l2NormList))))) * np.sign(np.mean(np.array(l2RookList)*np.abs(np.array(l2RookList)) - np.array(l2NormList)*np.abs(np.array(l2NormList))))) / totalRMSRookvNorm)
            #csv_out.append(np.sqrt(np.abs(np.mean(np.array(l2RookBList)**2) - np.mean(np.array(l2RookAltList)**2))) / totalRMSRookvRook)
            #csv_out.append(np.sqrt(np.abs(np.mean(np.array(l2RookList)**2) - np.mean(np.array(l2NormList)**2))) / totalRMSRookvNorm)
            csv_out.append(float(((np.array(l2NormBList) - np.array(l2NormAltList)) >0).sum()) / float((np.array(l2NormBList) - np.array(l2NormAltList)).shape[0]))
            csv_out.append(float(((np.array(l2RookBList) - np.array(l2RookAltList)) >0).sum()) / float((np.array(l2RookBList) - np.array(l2RookAltList)).shape[0]))
            csv_out.append(float(((np.array(l2RookList) - np.array(l2NormList)) >0).sum()) / float((np.array(l2RookList) - np.array(l2NormList)).shape[0]))
            csv_out.append(np.sqrt(np.mean(np.array(nnTrack)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(rrTrack)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(rnTrack)**2)))
            csv_out.append(np.mean(np.array(l2Norm3OnNN)))
            csv_out.append(np.mean(np.array(l2Rook3OnNN)))
            csv_out.append(np.mean(np.array(l2Norm4OnNN)))
            csv_out.append(np.mean(np.array(l2Norm3OnRR)))
            csv_out.append(np.mean(np.array(l2Rook3OnRR)))
            csv_out.append(np.mean(np.array(l2Norm4OnRR)))
            meanAvgRN3OnNN = (np.mean(np.array(l2Norm3OnNN)) + np.mean(np.array(l2Rook3OnNN))) / 2.0
            meanAvgNN4OnNN = (np.mean(np.array(l2Norm3OnNN)) + np.mean(np.array(l2Norm4OnNN))) / 2.0
            meanAvgRN3OnRR = (np.mean(np.array(l2Norm3OnRR)) + np.mean(np.array(l2Rook3OnRR))) / 2.0
            meanAvgNN4OnRR = (np.mean(np.array(l2Norm3OnRR)) + np.mean(np.array(l2Norm4OnRR))) / 2.0
            meanAvgN3OnNNVsRR = (np.mean(np.array(l2Norm3OnNN)) + np.mean(np.array(l2Norm3OnRR))) / 2.0
            meanAvgR3OnNNVsRR = (np.mean(np.array(l2Rook3OnNN)) + np.mean(np.array(l2Rook3OnRR))) / 2.0
            csv_out.append(meanAvgRN3OnNN)
            csv_out.append(meanAvgNN4OnNN)
            csv_out.append(meanAvgRN3OnRR)
            csv_out.append(meanAvgNN4OnRR)
            csv_out.append(meanAvgN3OnNNVsRR)
            csv_out.append(meanAvgR3OnNNVsRR)
            csv_out.append(np.std(np.array(l2Norm3OnNN)))
            csv_out.append(np.std(np.array(l2Rook3OnNN)))
            csv_out.append(np.std(np.array(l2Norm4OnNN)))
            csv_out.append(np.std(np.array(l2Norm3OnRR)))
            csv_out.append(np.std(np.array(l2Rook3OnRR)))
            csv_out.append(np.std(np.array(l2Norm4OnRR)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Norm3OnNN)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Rook3OnNN)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Norm4OnNN)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Norm3OnRR)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Rook3OnRR)**2)))
            csv_out.append(np.sqrt(np.mean(np.array(l2Norm4OnRR)**2)))
            csv_out.append((np.mean(np.array(l2Rook3OnNN)) - np.mean(np.array(l2Norm3OnNN))) / meanAvgRN3OnNN)
            csv_out.append((np.mean(np.array(l2Norm4OnNN)) - np.mean(np.array(l2Norm3OnNN))) / meanAvgNN4OnNN)
            csv_out.append((np.mean(np.array(l2Rook3OnRR)) - np.mean(np.array(l2Norm3OnRR))) / meanAvgRN3OnRR)
            csv_out.append((np.mean(np.array(l2Norm4OnRR)) - np.mean(np.array(l2Norm3OnRR))) / meanAvgNN4OnRR)
            csv_out.append((np.mean(np.array(l2Norm3OnRR)) - np.mean(np.array(l2Norm3OnNN))) / meanAvgN3OnNNVsRR)
            csv_out.append((np.mean(np.array(l2Rook3OnRR)) - np.mean(np.array(l2Rook3OnNN))) / meanAvgR3OnNNVsRR)
            if topInd == 0 and subInd == 0:
                with open('results/' + resultsFileName + 'results.csv', 'w') as f_handle:
                    np.savetxt(f_handle,np.array(csv_out)[np.newaxis,:])
            else:
                with open('results/' + resultsFileName + 'results.csv', 'a') as f_handle:
                    np.savetxt(f_handle,np.array(csv_out)[np.newaxis,:])
            del model, modelB, modelC
            del l2NormList, l2NormBList, l2RookList, l2NormAltList, l2RookAltList, l2RookBList, newTotalTrain, newTotalTarget, rookTotalTrain, rookTotalTarget, \
                l2Norm3OnNN, l2Norm3OnRR, l2Norm4OnNN, l2Norm4OnRR
            del fNormVal, fNormTar, fNormVal3, fNormVal4, fNormTar3, fbVal, fbTar, fRookVal, fRookTar, fRookValB, fRookTarB, normBNormComp, rookNormComp, rookBRookComp
            #print '- Validation Cell Complete -'
    l2NormList = np.hstack(l2NormListHolder)
    l2RookList = np.hstack(l2RookListHolder)
    l2Norm3OnNN = np.hstack(l2Norm3OnNNHolder)
    l2Norm4OnNN = np.hstack(l2Norm4OnNNHolder)
    l2Rook3OnNN = np.hstack(l2Rook3OnNNHolder)
    l2Norm3OnRR = np.hstack(l2Norm3OnRRHolder)
    l2Norm4OnRR = np.hstack(l2Norm4OnRRHolder)
    l2Rook3OnRR = np.hstack(l2Rook3OnRRHolder)
    l2NormAltList = np.hstack(l2NormAltListHolder)
    l2RookAltList = np.hstack(l2RookAltListHolder)
    l2NormBList = np.hstack(l2NormBListHolder)
    l2RookBList = np.hstack(l2RookBListHolder)
    np.save('results/' + resultsFileName + 'l2NormList.npy', l2NormList, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2RookList.npy', l2RookList, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Norm3OnNN.npy', l2Norm3OnNN, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Norm4OnNN.npy', l2Norm4OnNN, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Rook3OnNN.npy', l2Rook3OnNN, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Norm3OnRR.npy', l2Norm3OnRR, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Norm4OnRR.npy', l2Norm4OnRR, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2Rook3OnRR.npy', l2Rook3OnRR, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2NormBList.npy', l2NormBList, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2RookBList.npy', l2RookBList, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2NormAltList.npy', l2NormAltList, allow_pickle=True)
    np.save('results/' + resultsFileName + 'l2RookAltList.npy', l2RookAltList, allow_pickle=True)
    del fNorm, fRook, l2NormListHolder, l2RookListHolder, l2Norm3OnNNHolder, l2Rook3OnNNHolder, \
        l2Norm3OnRRHolder, l2Rook3OnRRHolder, l2NormBListHolder, l2RookBListHolder, \
        l2NormAltListHolder, l2RookAltListHolder, l2Norm4OnNNHolder, l2Norm4OnRRHolder
    del l2NormList, l2RookList, l2Norm3OnNN, l2Rook3OnNN, l2Norm3OnRR, l2Rook3OnRR, l2NormAltList, \
        l2RookAltList, l2NormBList, l2RookBList, l2Norm4OnNN, l2Norm4OnRR
    print('Final Save Complete')
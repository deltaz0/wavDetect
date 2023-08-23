import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import pickle as pickle
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv1D, TimeDistributed
import matplotlib.pyplot as plt
import os
from tensorflow import keras as K
from keras_visualizer import visualizer
import time

skipTrain = False

outFolder = './results/23modeldata/'
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)


# totalLen = 80000
totalLen = 161840
numFeatures = 32
minPeriod = 10
maxPeriod = 200
# filterSize = 32
filterSize = 64
# modelVersion = 'rollhalfseqbase32iters'
# modelVersionAlt = 'newbase20iters'
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

for modelVersion in ('rollhalfseqbase32iters', 'newbase20iters'):
    if modelVersion == 'newbase20iters':
        fullRoll = False
        rollAmt = 1
        trainIters = 20
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
        for topInd in range(1):
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
            for subInd in range(1):
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
                K.utils.plot_model(model, outFolder + modelVersion + '-' + str(0) + fileSuffix + '-plotmodel.png', rankdir='TB', show_shapes=True)
                with open(outFolder + modelVersion + '-' + str(0) + fileSuffix + '-summary.txt', "w") as outf:
                    model.summary(print_fn=lambda x: outf.write(x + '\n'))
                    outf.close()
                visualizer(model, file_name = outFolder + modelVersion + '-' + str(0) + fileSuffix + '-kerasvis', file_format='png', view=False)
                if not skipTrain:
                    start = time.time()
                    for i in range(3):
                        rI = np.random.randint(fTotalTrain.shape[0]//seqL)*seqL
                        xTrain = np.roll(fTotalTrain,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
                        yTrain = np.roll(fTotalTarget,rI,axis=0)[:seqL*(fTotalTrain.shape[0]//seqL)].reshape(-1,seqL,fTotalTrain.shape[1])
                        history = model.fit(xTrain,yTrain,batch_size=batchSize,shuffle=False,epochs=1)
                        # rnTrack = np.concatenate((rnTrack,history.history['loss']))
                    trainTime = (time.time()-start) / 3
                    print(trainTime)
                    np.save(outFolder + modelVersion + '-' + str(0) + fileSuffix + '-trainTime.npy', trainTime)
                    modelDict = {
                        'traintimeperepoch' : trainTime,
                        'trainshape' : fTotalTrain.shape,
                        'traintype' : fTotalTrain.dtype,
                        'batchsize' : batchSize,
                        'trainIters' : trainIters * 10
                        }  
                    with open(outFolder + modelVersion + '-' + str(0) + fileSuffix + '-trainStats.txt', 'w') as f: 
                        for key, value in modelDict.items(): 
                            f.write('%s: %s\n' % (key, value))
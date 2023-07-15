import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys
import csv
import pickle as pickle

maxs = np.zeros(128)
with open('myOutNewNonDecodedNew.csv', 'r') as csvfile:
    cList = csv.reader(csvfile)
    for i,o in enumerate(cList):
        maxs[i] = len(o)

for fileNum in range(20,21):
    csn = np.genfromtxt('nondecodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in range(20,21):
    csd = np.genfromtxt('nondecodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in range(21,40):
    csn = np.concatenate((csn,np.genfromtxt('nondecodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
for fileNum in range(21,40):
    csd = np.concatenate((csd,np.genfromtxt('nondecodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
maxs[maxs==1] += 1
csn = csn.astype('int')
csd = csd.astype('int')
csn = csn / ((maxs - 1).astype('float'))
csd = csd / ((maxs - 1).astype('float'))

packs = csn
packd = csd
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNew23NDNoMask.pkl', 'wb')
outputRook = open('featuresRookNew23NDNoMask.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()

ind = np.zeros((128,),bool)
maskIndices = [4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27]
ind[maskIndices] = True
packs = csn[:,ind]
packd = csd[:,ind]
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNew23NDMaskBase.pkl', 'wb')
outputRook = open('featuresRookNew23NDMaskBase.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()

ind = np.zeros((128,),bool)
maskIndices = [4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27]
ind[maskIndices] = True
packs = csn[:,ind]
packd = csd[:,ind]
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNew23NDMaskDeriv.pkl', 'wb')
outputRook = open('featuresRookNew23NDMaskDeriv.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()


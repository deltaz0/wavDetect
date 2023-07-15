import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys
import csv
import pickle as pickle

maxs = np.zeros(176)
with open('myOutNewNew.csv', 'r') as csvfile:
    cList = csv.reader(csvfile)
    for i,o in enumerate(cList):
        maxs[i] = len(o)

for fileNum in range(20,21):
    csn = np.genfromtxt('decodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in range(20,21):
    csd = np.genfromtxt('decodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in range(21,40):
    csn = np.concatenate((csn,np.genfromtxt('decodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
for fileNum in range(21,40):
    csd = np.concatenate((csd,np.genfromtxt('decodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
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
outputNorm = open('featuresNormalNew23DecNoMask.pkl', 'wb')
outputRook = open('featuresRookNew23DecNoMask.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()

ind = np.zeros((176,),bool)
maskIndices = [ 5, 7, 10, 11, 12, 13, 14, 16, 17, 18, 22, 48, 119, 120, 121, 123,
               126, 129, 130, 172, 173, 175 ]
ind[maskIndices] = True

packs = csn[:,ind]
packd = csd[:,ind]
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNew23DecMaskBase.pkl', 'wb')
outputRook = open('featuresRookNew23DecMaskBase.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()


ind = np.zeros((176,),bool)
maskIndices = [ 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 22, 48,
               37, 38, 42, 43, 45, 57, 58, 59, 60, 61, 63, 64, 87, 88, 89,
               90, 91, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
               106, 107, 108, 109, 110, 111, 112,
               115, 116, 117, 119, 120, 121, 122,
               123, 125, 126, 127, 129, 130, 172, 173, 175 ]
ind[maskIndices] = True

packs = csn[:,ind]
packd = csd[:,ind]
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNew23DecMaskWDeriv.pkl', 'wb')
outputRook = open('featuresRookNew23DecMaskWDeriv.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()



# coding: utf-8

# In[1]:


import numpy as np
import csv
import matplotlib.pyplot as plt
import random
import sys
import csv
import cPickle as pickle

maxs = np.zeros(128)
with open('../ncoderfork/Project3/myOutNewNonDecodedNew.csv', 'rb') as csvfile:
    cList = csv.reader(csvfile)
    for i,o in enumerate(cList):
        maxs[i] = len(o)
for fileNum in xrange(20,21):
    csn = np.genfromtxt('../ncoderfork/Project3/nondecodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in xrange(20,21):
    csd = np.genfromtxt('../ncoderfork/Project3/nondecodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")
for fileNum in xrange(21,40):
    csn = np.concatenate((csn,np.genfromtxt('../ncoderfork/Project3/nondecodedclassesNew/pn' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
for fileNum in xrange(21,40):
    csd = np.concatenate((csd,np.genfromtxt('../ncoderfork/Project3/nondecodedclassesNew/pd' + str(fileNum) + 'List.txt', delimiter=",")),axis=0)
maxs[maxs==1] += 1
csn = csn.astype('int')
csd = csd.astype('int')
csn = csn / ((maxs - 1).astype('float'))
csd = csd / ((maxs - 1).astype('float'))
#ind = np.ones((176,),bool)
#maskIndices = [3,8,9]
#ind[maskIndices] = False
packs = csn
packd = csd
packs = (packs * 2.0) - 1.0
packd = (packd * 2.0) - 1.0
if packd.shape[0] < packs.shape[0]:
    packs = packs[:packd.shape[0]]
elif packs.shape[0] < packd.shape[0]:
    packd = packd[:packs.shape[0]]
outputNorm = open('featuresNormalNonDecodedNew.pkl', 'wb')
outputRook = open('featuresRookNonDecodedNew.pkl', 'wb')
pickle.dump(packs,outputNorm)
pickle.dump(packd,outputRook)
outputNorm.close()
outputRook.close()



# coding: utf-8

# In[4]:


import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
system = 57779
newpath = "./" + str(system) + "23figs"
if not os.path.exists(newpath):
    os.makedirs(newpath)
runVer = 'newbase20iters-FakeToyPermuteRand-A'
resultsFileName = 'results/' + runVer + '/results.csv'
normListFileName = 'results/' + runVer + '/l2NormList.npy'
normList = np.load(normListFileName, allow_pickle=True)
normList = np.swapaxes(np.array(np.split(np.swapaxes(normList,0,1),16)),-2,-1)
results = np.genfromtxt(resultsFileName)
results = results[-16:,:]
print(' -- results[:,index] map --                  \n' +
      ' 1st dim = run version                                           \n' +
      ' 4 full database runs, each with .25 of data and rnd start index \n' +
      ' so results[9,:] = 3rd dataset sweep, 2nd quarter of data        \n' +
      'norm b    - norm 2 on N1N2 model             \n' +
      'norm alt  - norm 1 on N1N2 model             \n' +
      'rook b    - rook 2 on R1R2 model             \n' +
      'rook alt  - rook 1 on R1R2 model             \n' +
      'rook      - rook 1 on N1R1 model             \n' +
      'norm      - norm 1 on N1R1 model             \n' +
      'norm3onNN - rook 3 (extra) on N1N2 model     \n' +
      'norm4onNN - norm 4 (extra) on N1N2 model     \n' +
      'rook3onNN - norm 3 (extra) on N1N2 model     \n' +
      'norm3onRR - rook 3 (extra) on R1R2 model     \n' +
      'norm4onRR - norm 4 (extra) on R1R2 model     \n' +
      'rook3onRR - norm 3 (extra) on R1R2 model     \n' +
      '0    - overall mean error: norm b           \n' +
      '1    - overall mean error: norm alt         \n' +
      '2    - overall mean error: rook b           \n' +
      '3    - overall mean error: rook alt         \n' +
      '4    - overall mean error: rook             \n' +
      '5    - overall mean error: norm             \n' +
      '6    - avg overall mean across norm & norm           \n' +
      '7    - avg overall mean across rook & rook           \n' +
      '8    - avg overall mean across rook & norm           \n' +
      '9    - overall std of error: norm b           \n' +
      '10   - overall std of error: norm alt         \n' +
      '11   - overall std of error: rook b           \n' +
      '12   - overall std of error: rook alt         \n' +
      '13   - overall std of error: rook             \n' +
      '14   - overall std of error: norm             \n' +
      '15   - overall rms of error: norm b           \n' +
      '16   - overall rms of error: norm alt         \n' +
      '17   - overall rms of error: rook b           \n' +
      '18   - overall rms of error: rook alt         \n' +
      '19   - overall rms of error: rook             \n' +
      '20   - overall rms of error: norm             \n' +
      '21   - avg overall rms across norm & norm           \n' +
      '22   - avg overall rms across rook & rook           \n' +
      '23   - avg overall rms across rook & norm           \n' +
      '24   - (mean error diff normb - normalt) / avg NN mean   \n' +
      '25   - (mean error diff rookb - rookalt) / avg RR mean   \n' +
      '26   - (mean error diff rook - norm) / avg RN mean   \n' +
      '27   - (rms error diff normb - normalt) / avg NN rms   \n' +
      '28   - (rms error diff rookb - rookalt) / avg RR rms   \n' +
      '29   - (rms error diff rook - norm) / avg RN rms   \n' +
      '30   - sign-preserve (rms error diff normb - normalt) / avg NN rms \n' +
      '31   - sign-preserve (rms error diff rookb - rookalt) / avg RR rms \n' +
      '32   - sign-preserve (rms error diff rook - norm) / avg RN rms     \n' +
      '33   - binary comp: (num where normb > normalt) / shape[0] \n' +
      '34   - binary comp: (num where rookb > rookalt) / shape[0] \n' +
      '35   - binary comp: (num where rook > norm) / shape[0] \n' +
      '36   - rms of NN loss across training iterations        \n' +
      '37   - rms of RR loss across training iterations        \n' +
      '38   - rms of RN loss across training iterations        \n' +
      '39   - overall mean error: norm 3 on NN           \n' +
      '40   - overall mean error: rook 3 on NN           \n' +
      '41   - overall mean error: norm 4 on NN           \n' +
      '42   - overall mean error: norm 3 on RR           \n' +
      '43   - overall mean error: rook 3 on RR           \n' +
      '44   - overall mean error: norm 4 on RR           \n' +
      '45   - avg overall mean across norm3onNN & rook3onNN \n' +
      '46   - avg overall mean across norm3onNN & norm4onNN \n' +
      '47   - avg overall mean across norm3onRR & rook3onRR \n' +
      '48   - avg overall mean across norm3onRR & norm4onRR \n' +
      '49   - avg overall mean across norm3onNN & norm3onRR \n' +
      '50   - avg overall mean across rook3onNN & rook3onRR \n' +
      '51   - overall std of error: norm 3 on NN           \n' +
      '52   - overall std of error: rook 3 on NN           \n' +
      '53   - overall std of error: norm 4 on NN           \n' +
      '54   - overall std of error: norm 3 on RR           \n' +
      '55   - overall std of error: rook 3 on RR           \n' +
      '56   - overall std of error: norm 4 on RR           \n' +
      '57   - overall rms of error: norm 3 on NN           \n' +
      '58   - overall rms of error: rook 3 on NN           \n' +
      '59   - overall rms of error: norm 4 on NN           \n' +
      '60   - overall rms of error: norm 3 on RR           \n' +
      '61   - overall rms of error: rook 3 on RR           \n' +
      '62   - overall rms of error: norm 4 on RR           \n' +
      '63   - (mean error diff: rook3onNN - norm3onNN) / avgmean r3n3onNN \n' +
      '64   - (mean error diff: norm4onNN - norm3onNN) / avgmean n4n3onNN \n' +
      '65   - (mean error diff: rook3onRR - norm3onRR) / avgmean r3n3onRR \n' +
      '66   - (mean error diff: norm4onRR - norm3onRR) / avgmean n4n3onRR \n' +
      '67   - (mean error diff: norm3onRR - norm3onNN) / avgmean n3onNNRR \n' +
      '68   - (mean error diff: rook3onRR - rook3onNN) / avgmean r3onNNRR \n' +
      ' -- done -- ')



print(results.shape)




gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,0],label='Clean2 on ModelClean')
plt.plot(results[:,1],label='Clean1 on ModelClean')
plt.plot(results[:,2],label='Altered2 on ModelAltered')
plt.plot(results[:,3],label='Altered1 on ModelAltered')
plt.plot(results[:,4],label='Altered1 on ModelCombined')
plt.plot(results[:,5],label='Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Mean Prediction Error")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,0]),label='Clean2 on ModelClean')
plt.bar(1,np.mean(results[:,1]),label='Clean1 on ModelClean')
plt.bar(2,np.mean(results[:,2]),label='Altered2 on ModelAltered')
plt.bar(3,np.mean(results[:,3]),label='Altered1 on ModelAltered')
plt.bar(4,np.mean(results[:,4]),label='Altered1 on ModelCombined')
plt.bar(5,np.mean(results[:,5]),label='Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()



gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,6],label='Clean1 & Clean2 on ModelClean')
plt.plot(results[:,7],label='Altered1 & Altered2 on ModelAltered')
plt.plot(results[:,8],label='Clean1 & Altered1 on ModelCombined')
plt.legend()
plt.suptitle("Overall Mean Prediction Error Across All Data per Model")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,6]),label='Clean1 & Clean2 on ModelClean')
plt.bar(1,np.mean(results[:,7]),label='Altered1 & Altered2 on ModelAltered')
plt.bar(2,np.mean(results[:,8]),label='Clean1 & Altered1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()
# aNvN = (np.mean(np.array(results[0])) + np.mean(np.array(results[1]))) / 2.0
# aRvR = (np.mean(np.array(results[2])) + np.mean(np.array(results[3]))) / 2.0
# aRvN = (np.mean(np.array(results[4])) + np.mean(np.array(results[5]))) / 2.0
# plt.plot(np.divide(results[:,1] - results[:,0],aNvN),label='NormB & NormAlt')
# plt.plot(np.divide(results[:,3] - results[:,2],aRvR),label='RookB & RookAlt')
# plt.plot(np.divide(results[:,4] - results[:,5],aRvN),label='Rook & Norm')


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,24],label='Clean2 vs Clean1 on ModelClean')
plt.plot(results[:,25],label='Altered2 vs Altered1 on ModelAltered')
plt.plot(results[:,26],label='Altered1 vs Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Mean Prediction Error Difference (Normalized by Total Mean Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,24]),label='Clean2 vs Clean1 on ModelClean')
plt.bar(1,np.mean(results[:,25]),label='Altered2 vs Altered1 on ModelAltered')
plt.bar(2,np.mean(results[:,26]),label='Altered1 vs Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()
# plt.plot(np.divide(results[:,1] - results[:,0],results[:,6]),label='$c_{K_i}$')
# plt.plot(np.divide(results[:,3] - results[:,2],results[:,7]),label='$c_{U_i}$')
# plt.plot(np.divide(results[:,4] - results[:,5],results[:,8]),label='$c_{H_i}$')
# plt.legend()
# plt.gcf().set_size_inches(13,7)
# plt.suptitle("Avg Means")
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.gcf().set_size_inches(13,7)
# plt.suptitle("Scaled Mean Comparisons (unaltered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + '23figs/method2Alt.png')
# plt.ylabel('Value')
# plt.xlabel('Model Version')
# plt.tight_layout()
# plt.show()
# plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(np.divide(results[:,0],results[:,6]),label='Clean2 on ModelClean')
plt.plot(np.divide(results[:,1],results[:,6]),label='Clean1 on ModelClean')
plt.plot(np.divide(results[:,2],results[:,7]),label='Altered2 on ModelAltered')
plt.plot(np.divide(results[:,3],results[:,7]),label='Altered1 on ModelAltered')
plt.plot(np.divide(results[:,4],results[:,8]),label='Altered1 on ModelCombined')
plt.plot(np.divide(results[:,5],results[:,8]),label='Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Mean Prediction Error (Normalized by Total Mean Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(np.divide(results[:,0],results[:,6])),label='Clean2 on ModelClean')
plt.bar(1,np.mean(np.divide(results[:,1],results[:,6])),label='Clean1 on ModelClean')
plt.bar(2,np.mean(np.divide(results[:,2],results[:,7])),label='Altered2 on ModelAltered')
plt.bar(3,np.mean(np.divide(results[:,3],results[:,7])),label='Altered1 on ModelAltered')
plt.bar(4,np.mean(np.divide(results[:,4],results[:,8])),label='Altered1 on ModelCombined')
plt.bar(5,np.mean(np.divide(results[:,5],results[:,8])),label='Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,9],label='Clean2 on ModelClean')
plt.plot(results[:,10],label='Clean1 on ModelClean')
plt.plot(results[:,11],label='Altered2 on ModelAltered')
plt.plot(results[:,12],label='Altered1 on ModelAltered')
plt.plot(results[:,13],label='Altered1 on ModelCombined')
plt.plot(results[:,14],label='Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Standard Dev of Prediction Error")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,9]),label='Clean2 on ModelClean')
plt.bar(1,np.mean(results[:,10]),label='Clean1 on ModelClean')
plt.bar(2,np.mean(results[:,11]),label='Altered2 on ModelAltered')
plt.bar(3,np.mean(results[:,12]),label='Altered1 on ModelAltered')
plt.bar(4,np.mean(results[:,13]),label='Altered1 on ModelCombined')
plt.bar(5,np.mean(results[:,14]),label='Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,15],label='Clean2 on ModelClean')
plt.plot(results[:,16],label='Clean1 on ModelClean')
plt.plot(results[:,17],label='Altered2 on ModelAltered')
plt.plot(results[:,18],label='Altered1 on ModelAltered')
plt.plot(results[:,19],label='Altered1 on ModelCombined')
plt.plot(results[:,20],label='Clean1 on ModelCombined')
plt.legend()
plt.suptitle("RMS of Prediction Error")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,15]),label='Clean2 on ModelClean')
plt.bar(1,np.mean(results[:,16]),label='Clean1 on ModelClean')
plt.bar(2,np.mean(results[:,17]),label='Altered2 on ModelAltered')
plt.bar(3,np.mean(results[:,18]),label='Altered1 on ModelAltered')
plt.bar(4,np.mean(results[:,19]),label='Altered1 on ModelCombined')
plt.bar(5,np.mean(results[:,20]),label='Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,21],label='Clean1 & Clean2 on ModelClean')
plt.plot(results[:,22],label='Altered1 & Altered2 on ModelAltered')
plt.plot(results[:,23],label='Clean1 & Altered1 on ModelCombined')
plt.legend()
plt.suptitle("Overall RMS of Prediction Error Across All Data per Model")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,21]),label='Clean1 & Clean2 on ModelClean')
plt.bar(1,np.mean(results[:,22]),label='Altered1 & Altered2 on ModelAltered')
plt.bar(2,np.mean(results[:,23]),label='Clean1 & Altered1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(np.divide(results[:,15],results[:,21]),label='Clean2 on ModelClean')
plt.plot(np.divide(results[:,16],results[:,21]),label='Clean1 on ModelClean')
plt.plot(np.divide(results[:,17],results[:,22]),label='Altered2 on ModelAltered')
plt.plot(np.divide(results[:,18],results[:,22]),label='Altered1 on ModelAltered')
plt.plot(np.divide(results[:,19],results[:,23]),label='Altered1 on ModelCombined')
plt.plot(np.divide(results[:,20],results[:,23]),label='Clean1 on ModelCombined')
plt.legend()
plt.suptitle("RMS of Prediction Error (Normalized by Total RMS Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(np.divide(results[:,15],results[:,21])),label='Clean2 on ModelClean')
plt.bar(1,np.mean(np.divide(results[:,16],results[:,21])),label='Clean1 on ModelClean')
plt.bar(2,np.mean(np.divide(results[:,17],results[:,22])),label='Altered2 on ModelAltered')
plt.bar(3,np.mean(np.divide(results[:,18],results[:,22])),label='Altered1 on ModelAltered')
plt.bar(4,np.mean(np.divide(results[:,19],results[:,23])),label='Altered1 on ModelCombined')
plt.bar(5,np.mean(np.divide(results[:,20],results[:,23])),label='Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()
# plt.plot(results[:,26],label='$c_{H_i}$')
# plt.plot(results[:,24],label='$c_{K_i}$')
# plt.plot(results[:,25],label='$c_{U_i}$')
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (altered case)")
# plt.ylabel("Mean difference scaled by mean avg")
# plt.xlabel("Iteration")
# plt.ylim(-0.05,0.2)
# plt.savefig(str(system) + "23figs/diff3Comp.png")
# plt.show()
# plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,27],label='Clean2 vs Clean1 on ModelClean')
plt.plot(results[:,28],label='Altered2 vs Altered1 on ModelAltered')
plt.plot(results[:,29],label='Altered1 vs Clean1 on ModelCombined')
plt.legend()
plt.suptitle("RMS of Prediction Error Difference (Normalized by Total RMS Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,27]),label='Clean2 vs Clean1 on ModelClean')
plt.bar(1,np.mean(results[:,28]),label='Altered2 vs Altered1 on ModelAltered')
plt.bar(2,np.mean(results[:,29]),label='Altered1 vs Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,30],label='Clean2 vs Clean1 on ModelClean')
plt.plot(results[:,31],label='Altered2 vs Altered1 on ModelAltered')
plt.plot(results[:,32],label='Altered1 vs Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Sign-Preserving RMS of Prediction Error Difference (Normalized by Total RMS Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,30]),label='Clean2 vs Clean1 on ModelClean')
plt.bar(1,np.mean(results[:,31]),label='Altered2 vs Altered1 on ModelAltered')
plt.bar(2,np.mean(results[:,32]),label='Altered1 vs Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()



gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,33]/normList.shape[-1],label='Clean2 vs Clean1 on ModelClean')
plt.plot(results[:,34]/normList.shape[-1],label='Altered2 vs Altered1 on ModelAltered')
plt.plot(results[:,35]/normList.shape[-1],label='Altered1 vs Clean1 on ModelCombined')
plt.legend()
plt.suptitle("Binary Comparisons (Overall Percentage where 1st Dataset has Higher Error than 2nd Dataset)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,33]/normList.shape[-1]),label='Clean2 vs Clean1 on ModelClean')
plt.bar(1,np.mean(results[:,34]/normList.shape[-1]),label='Altered2 vs Altered1 on ModelAltered')
plt.bar(2,np.mean(results[:,35]/normList.shape[-1]),label='Altered1 vs Clean1 on ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()


gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:,36],label='ModelClean')
plt.plot(results[:,37],label='ModelAltered')
plt.plot(results[:,38],label='ModelCombined')
plt.legend()
plt.suptitle("RMS of Total Model Loss History Comparisons")
plt.ylabel('Loss')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:,36]),label='ModelClean')
plt.bar(1,np.mean(results[:,37]),label='ModelAltered')
plt.bar(2,np.mean(results[:,38]),label='ModelCombined')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()

gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1]) 
plt.subplot(gs[0])
plt.plot(results[:, 64], label='Clean4 vs Clean3 on ModelClean')
plt.plot(results[:, 63], label='Altered3 vs Clean3 on ModelClean')
plt.legend()
plt.suptitle("Mean Prediction Error Difference (Normalized by Total Mean Across All Data per Model)")
plt.ylabel('Value')
plt.xlabel('Model Version')
plt.subplot(gs[1])
plt.bar(0,np.mean(results[:, 64]), label='Clean4 vs Clean3 on ModelClean')
plt.bar(1,np.mean(results[:, 63]), label='Altered3 vs Clean3 on ModelClean')
plt.setp(plt.gca(), ylim=plt.gcf().get_axes()[0].get_ylim())
plt.gca().get_xaxis().set_visible(False)
plt.ylabel("Avg Across Versions")
for line in plt.gcf().get_axes()[0].legend().get_lines():
    line.set_linewidth(2.5)
plt.gcf().set_size_inches(13,7)
plt.tight_layout()
plt.show()
plt.clf()

# plt.plot(np.divide(results[:, 39] - results[:, 41], results[:, 46]), label='$c_{k_i}$ $/$ $a_{k_i}$')
# plt.plot(np.divide(results[:, 40] - results[:, 41], results[:, 45]), label='$c_{u_i}$ $/$ $a_{u_i}$')
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (unaltered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + '23figs/method1Unalt.png')
# plt.show()
# plt.clf()

# system = 57779
# newpath = "./" + str(system) + "23figs"
# if not os.path.exists(newpath):
#     os.makedirs(newpath)
# # resultsFileName = 'results/base20iters-New23DecMaskDeriv-A/results.csv'
# results = np.genfromtxt(resultsFileName)
# results = results[-16:,:]
# print(results.shape)

# plt.plot(np.divide(results[:,1] - results[:,0],results[:,6]),label='$c_{K_i}$')
# plt.plot(np.divide(results[:,3] - results[:,2],results[:,7]),label='$c_{U_i}$')
# plt.plot(np.divide(results[:,4] - results[:,5],results[:,8]),label='$c_{H_i}$')
# plt.legend()
# plt.suptitle("Avg Means")
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (altered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + '23figs/method2Alt.png')
# plt.show()
# plt.clf()

# plt.plot(np.divide(results[:, 39] - results[:, 41], results[:, 46]), label='$c_{k_i}$ $/$ $a_{k_i}$')
# plt.plot(np.divide(results[:, 40] - results[:, 41], results[:, 45]), label='$c_{u_i}$ $/$ $a_{u_i}$')
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (altered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + '23figs/method1Alt.png')
# plt.show()
# plt.clf()

# plt.plot(results[:,39],label='Norm3 on NN')
# plt.plot(results[:,40],label='Rook3 on NN')
# plt.plot(results[:,41],label='Norm4 on NN')
# plt.plot(results[:,42],label='Norm3 on RR')
# plt.plot(results[:,43],label='Rook3 on RR')
# plt.plot(results[:,44],label='Norm4 on RR')
# plt.plot(results[:,69],label='Norm5 on NN')
# plt.legend()
# plt.suptitle("Means")
# plt.show()
# plt.clf()
# plt.plot(results[:,45],label='Rook3 & Norm3 on NN')
# plt.plot(results[:,46],label='Norm4 & Norm3 on NN')
# plt.plot(results[:,47],label='Rook3 & Norm3 on RR')
# plt.plot(results[:,48],label='Norm4 & Norm3 on RR')
# plt.plot(results[:,70],label='Norm5 & Norm3 on NN')
# plt.legend()
# plt.suptitle("Avg Means")
# plt.show()
# plt.clf()
# plt.plot(results[:,49],label='Norm3 on NN & RR')
# plt.plot(results[:,50],label='Rook3 on NN & RR')
# plt.legend()
# plt.suptitle("Avg Means")
# plt.show()
# plt.clf()
# plt.plot(results[:,51],label='Norm3 on NN')
# plt.plot(results[:,52],label='Rook3 on NN')
# plt.plot(results[:,53],label='Norm4 on NN')
# plt.plot(results[:,54],label='Norm3 on RR')
# plt.plot(results[:,55],label='Rook3 on RR')
# plt.plot(results[:,56],label='Norm4 on RR')
# plt.plot(results[:,71],label='Norm5 on NN')
# plt.legend()
# plt.suptitle("Std Devs")
# plt.show()
# plt.clf()
# plt.plot(results[:,57],label='Norm3 on NN')
# plt.plot(results[:,58],label='Rook3 on NN')
# plt.plot(results[:,59],label='Norm4 on NN')
# plt.plot(results[:,60],label='Norm3 on RR')
# plt.plot(results[:,61],label='Rook3 on RR')
# plt.plot(results[:,62],label='Norm4 on RR')
# plt.plot(results[:,72],label='Norm5 on NN')
# plt.legend()
# plt.suptitle("RMS's")
# plt.show()
# plt.clf()
# plt.plot(results[:,63],label='Rook3 vs Norm3 on NN')
# plt.plot(results[:,64],label='Norm4 vs Norm3 on NN')
# plt.plot(results[:,65],label='Rook3 vs Norm3 on RR')
# plt.plot(results[:,66],label='Norm4 vs Norm3 on RR')
# plt.plot(results[:,73],label='Norm5 vs Norm3 on NN')
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons")
# plt.show()
# plt.clf()
# plt.plot(results[:,67],label='Norm3 on NN vs RR')
# plt.plot(results[:,68],label='Rook3 on NN vs RR')
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons")
# plt.show()
# plt.clf()
# plt.plot(results[:,64],label='$c_{k_i}$ $/$ $a_{k_i}$ from CNN trained on $X_{train}$')
# plt.plot(results[:,63],label='$c_{u_i}$ $/$ $a_{u_i}$ from CNN trained on $X_{train}$')
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (altered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + 'figs/SMC-rvnNN-nvnNN-alt.png')
# plt.show()
# plt.clf()
# plt.plot(results[:,64],label='$c_{k_i}$ $/$ $a_{k_i}$ from CNN trained on $X_{train}$')
# plt.plot(results[:,73],label='$c_{u_i}$ $/$ $a_{u_i}$ from CNN trained on $X_{train}$')
# plt.ylim(-0.05,0.2)
# plt.legend()
# plt.suptitle("Scaled Mean Comparisons (unaltered case)")
# plt.ylabel('Mean Difference Scaled by Mean Avg')
# plt.xlabel('Iteration')
# plt.savefig(str(system) + 'figs/SMC-rvnNN-nvnNN-unalt.png')
# plt.show()
# plt.clf()


# In[ ]:





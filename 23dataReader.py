import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import os
system = 57779
newpath = "./" + str(system) + "23figs"
if not os.path.exists(newpath):
    os.makedirs(newpath)
modelVer    =   'newbase20iters'
modelType   =   'FakeToyPermuteRand-A'
for modelVer in [
    'newbase20iters',
    'rollhalfseqbase32iters'
]:
    for modelType in [
            'New23NDNoMask-A',
            'New23NDMaskBase-A',
            'New23NDMaskDeriv-A',
            'New23DecNoMask-A',
            'New23DecMaskBase-A',
            'New23DecMaskDeriv-A',
            'FakeToyPermuteRand-A',
            'FakeToyPermuteSin-A'
    ]:
        runVer = modelVer + '-' + modelType
        resultsFileName = 'results/' + runVer + '/results.csv'
        normListFileName = 'results/' + runVer + '/l2NormList.npy'
        resOutput = 'results/' + runVer + '/'
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

        outDict = {}

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
        # plt.show()
        plt.savefig(resOutput + 'MPE' + '.png')
        plt.clf()

        outDict.update({
            'MPEnormb'      : np.mean(results[:,0]),
            'MPEnormalt'    : np.mean(results[:,1]),
            'MPErookb'      : np.mean(results[:,2]),
            'MPErookalt'    : np.mean(results[:,3]),
            'MPErook'       : np.mean(results[:,4]),
            'MPEnorm'       : np.mean(results[:,5])    
        })

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
        # plt.show()
        plt.savefig(resOutput + 'TotalMPE' + '.png')
        plt.clf()

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
        # plt.show()
        plt.savefig(resOutput + 'scaledMPEDiff' + '.png')
        plt.clf()

        outDict.update({
            'scaledMPEDiffNormbnormalt'     : np.mean(results[:,24]),
            'scaledMPEDiffRookbrookalt'    : np.mean(results[:,25]),
            'scaledMPEDiffRooknorm'        : np.mean(results[:,26])    
        })

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
        # plt.show()
        plt.savefig(resOutput + 'scaledMPE' + '.png')
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
        # plt.show()
        plt.savefig(resOutput + 'StdPE' + '.png')
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
        # plt.show()
        plt.savefig(resOutput + 'RmsPE' + '.png')
        plt.clf()

        outDict.update({
            'rmsPEnormb'      : np.mean(results[:,15]),
            'rmsPEnormalt'    : np.mean(results[:,16]),
            'rmsPErookb'      : np.mean(results[:,17]),
            'rmsPErookalt'    : np.mean(results[:,18]),
            'rmsPErook'       : np.mean(results[:,19]),
            'rmsPEnorm'       : np.mean(results[:,20])    
        })

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
        # plt.show()
        plt.savefig(resOutput + 'TotalRmsPE' + '.png')
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
        # plt.show()
        plt.savefig(resOutput + 'ScaledRmsPE' + '.png')
        plt.clf()

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
        # plt.show()
        plt.savefig(resOutput + 'ScaledRmsPEDiff' + '.png')
        plt.clf()

        outDict.update({
            'scaledRmsPEDiffNormbnormalt'     : np.mean(results[:,27]),
            'scaledRmsPEDiffRookbrookalt'    : np.mean(results[:,28]),
            'scaledRmsPEDiffRooknorm'        : np.mean(results[:,29])    
        })

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
        # plt.show()
        plt.savefig(resOutput + 'SignPreserveScaledRmsPEDiff' + '.png')
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
        # plt.show()
        plt.savefig(resOutput + 'BinaryComparsionPE' + '.png')
        plt.clf()

        outDict.update({
            'BinComparisonNormbnormalt'     : np.mean(results[:,33]/normList.shape[-1]),
            'BinComparisonRookbrookalt'     : np.mean(results[:,34]/normList.shape[-1]),
            'BinComparisonRooknorm'         : np.mean(results[:,35]/normList.shape[-1])    
        })

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
        # plt.show()
        plt.savefig(resOutput + 'RmsTotalLoss' + '.png')
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
        # plt.show()
        plt.savefig(resOutput + 'scaledMPEDiff-3-4' + '.png')
        plt.clf()

        outDict.update({
            'scaledMPEDiffnorm4NNvnorm3NN'  : np.mean(results[:, 64]),
            'scaledMPEDiffrook3NNvnorm3NN'  : np.mean(results[:, 63]) 
        })

        with open(resOutput + 'resultsDict', 'wb') as handle:
            pickle.dump(outDict, handle)
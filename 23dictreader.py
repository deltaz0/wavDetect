import pickle
import csv
import numpy as np


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
        if modelVer == 'newbase20iters':
            modelPre = 'Base Arch, predict t+1, 200 Iters'
        else:
            modelPre = 'Base Arch, predict t+100, 320 Iters'
        if modelType == 'New23NDNoMask-A':
            modelSuf = 'Non-Decoded, No Mask'
        elif modelType == 'New23NDMaskBase-A':
            modelSuf = 'Non-Decoded, Diversity Mask'
        elif modelType == 'New23NDMaskDeriv-A':
            modelSuf = 'Non-Decoded, Diversity+Movement Mask'
        elif modelType == 'New23DecNoMask-A':
            modelSuf = 'Decoded, No Mask'
        elif modelType == 'New23DecMaskBase-A':
            modelSuf = 'Decoded, Diversity Mask'
        elif modelType == 'New23DecMaskDeriv-A':
            modelSuf = 'Decoded, Diversity+Movement Mask'
        elif modelType == 'FakeToyPermuteRand-A':
            modelSuf = 'Generated Signal Sim, Uniform Random Mutations'
        else:
            modelSuf = 'Generated Signal Sim, Random Sin-Based Mutations'
        runVer = modelVer + '-' + modelType
        resultsFileName = 'results/' + runVer + '/results.csv'
        normListFileName = 'results/' + runVer + '/l2NormList.npy'
        resOutput = 'results/' + runVer + '/'
        with open(resOutput + 'resultsDict', 'rb') as handle:
            loadDict = pickle.load(handle)
            print(' - test - ')

        with open(resOutput + 'resultsCSVDict.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for item in loadDict.items():
                writer.writerow(item)
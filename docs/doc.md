# Documention for nCoder

Acquired network packets captures from the approach described in [this paper](https://dl.acm.org/citation.cfm?id=2808141), both with the method being used and normal game traffic. Define 'normal' traffic as unaltered game packets and 'rook' traffic as packets which have Rook information smuggled inside them. Question is can we detect the use of such a system with only the raw data?

Project tries to solve it using conv nets and a unique comparison method. The ncoderfork/ folder contains an older c++ and python project which did some preprocessing on the data trying to parse out the protocol. The results from that project are saved as .pkl in ncoderdata/. One approach compares the rook v normal packet data, the other simulates the problem by trying to detect random data inserted into a signal composed of various waveforms. Going forward we will refer to the altered and unaltered simulated signals and rook and normal data, respectively.

Dilated multi-layer convnets are created and trained to predict the next value in the sequence for rook and normal data (one network for each, and a third for a combination of the two). Then these networks are evaluated on new data in different ways and different functions of the prediction accuracies can be used to detect if the test data is rook or normal data.

## Files

- Untitled.py - Earlier version of Untitled1.py.
- Untitled1.py - Earlier version of Untitled2.py.
- Untitled2.py - Loads data from featuresNormal(/Rook)NonDecodedExtract.pkl and applies our method to try and detect data alteration and saves relevant data to results/#####/results#.csv. (Probably earlier method than Untitled3/4.py)
- Untitled3.py -  Ealier version of Untitled4.py.
- Untitled4.py - Generates simulation of Rook and normal packet data using a combination of various waveforms and then inserting random permutations into half. Applies our method to try and detect the data alteration and saves relevant data to results/#####/results#.csv.
- dataPickler.py - Loads decoded (partially decoded protocol by manual pattern matching) packet data (decodedclasses/) from ncoderfork/, both rook and norm versions, and saves it to featuresNormal.pkl and featuresRook.pkl.
- dataPicklerAlt.py - Same as dataPickler.py, but pulls from decodedclassesNew/ and saves to featuresNormalNew.pkl and featuresRookNew.pkl.
- dataReader.py - Script which graphs the data saved by Untitled#.py, pulls from #####system/results#.csv.
- detector.py - Early version of convnet rook v norm detector, operates on featuresNormal.pkl and featuresRook.pkl.
- detectorAlt.py - Alternative to detector.py.
- detectorNonDecoded.py - Alternative to detector.py, further along, and operates on featuresNormalNonDecodedExtract.pkl and the Rook version.
- modelGrapher.py - Scratch script to save an image of the model.
- nonDecodedPickler.py - Loads non-decoded packet data (nondecodedclassesNew/) from ncoderfork/, both rook and norm versions, and saves it to featuresNormalNonDecodedNew.pkl and featuresRookNonDecodedNew.pkl.
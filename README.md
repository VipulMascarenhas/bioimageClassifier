# BioImage Classifier

Most proteins localize to specific regions where they perform their biological function. Fluorescent microscopy can reveal the subcellular localization patterns of tagged proteins.  The goal of this project is to use active learning to build a classifier that capable of classifying bioimages (encoded as feature vectors) according to subcellular localization patterns. 

## Learning Tasks

There are three data pools:

- Easy: A low-noise data pool

- Moderate: This pool has some noise (labels and features)

- Difficult: The points in this pool have a larger number of features than those in the easy and moderate pools. Some of these features are irrelevant.  Your algorithm will need to perform active learning and feature selection.

Each data pool consists of 4120 training images and 1000 test images.  Each image is represented as a feature vector (you do not need to do feature extraction yourself). There are 8 subcellular localization patterns: (i) Endosomes; (ii)  Lysosomes; (iii) Mitochondria; (iv) Peroxisomes; (v) Actin; (vi) Plasma Membrane; (vii) Microtubules; and (viii) Endoplasmic Reticulum.  The data are based on those released by Dr. Nicholas Hamilton for his paper Statistical and visual differentiation of high throughput subcellular imaging, N. Hamilton, J. Wang, M.C. Kerr and R.D. Teasdale, BMC Bioinformatics 2009, 10:94.  

Select and implement a suitable active learning algorithm and apply it to the training data. Additionally, implement a random learner that selects random images in the training data.  Using a budget of 2,500 calls to the oracle, compute and plot the test errors for each algorithm as a function of the number of calls to the oracle. Use the test data to compute the test errors.  Repeat this for the easy and moderate data pools.  If you are working on a team, or want extra credit, apply your algorithm to the difficult pool as well. 


### Blinded Predictions

For each data pool, use your active learning algorithm train a model using 2,500 labeled examples from the training data.  Use those models to make predictions for the data in the files named: "EASY_BLINDED.csv", "MODERATE_BLINDED.csv", and "DIFFICULT_BLINDED.csv".  The first column of those files is an instance ID.  You should create prediction files named "EASY_BLINDED_PRED.csv", "MODERATE_BLINDED_PRED.csv", and "DIFFICULT_BLINDED_PRED.csv". Each prediction should be on a different line.  Each line should have the following format:

instance_id, prediction



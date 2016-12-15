# BioImage Classifier

Most proteins localize to specific regions where they perform their biological function. Fluorescent microscopy can reveal the subcellular localization patterns of tagged proteins.  The goal of this project is to use active learning to build a classifier that capable of classifying bioimages (encoded as feature vectors) according to subcellular localization patterns. 

## Learning Tasks

There are three data pools:

- Easy: A low-noise data pool

- Moderate: This pool has some noise (labels and features)

- Difficult: The points in this pool have a larger number of features than those in the easy and moderate pools. Some of these features are irrelevant.

Each data pool consists of 4120 training images and 1000 test images.  Each image is represented as a feature vector. There are 8 subcellular localization patterns: (i) Endosomes; (ii)  Lysosomes; (iii) Mitochondria; (iv) Peroxisomes; (v) Actin; (vi) Plasma Membrane; (vii) Microtubules; and (viii) Endoplasmic Reticulum.  The data are based on those released by Dr. Nicholas Hamilton for his paper Statistical and visual differentiation of high throughput subcellular imaging, N. Hamilton, J. Wang, M.C. Kerr and R.D. Teasdale, BMC Bioinformatics 2009, 10:94.  

We implement an active learning algorithm and apply it to the training data. Additionally, we implement a random learner that selects random images in the training data.  Using a budget of 2,500 calls to the oracle, we compute and plot the test errors for each algorithm as a function of the number of calls to the oracle. The test data is used to compute the test errors.


### Blinded Predictions

For each data pool, we use our active learning algorithm train a model using 2,500 labeled examples from the training data. We then choose the best model to make predictions for the data in the files named: "EASY_BLINDED.csv", "MODERATE_BLINDED.csv", and "DIFFICULT_BLINDED.csv".  The first column of these files is an instance ID. Each prediction is stored in a file with the same name, but ending with 'PRED' and each line has the following format:
instance_id, prediction

### Approach:

We implement a [pool based active learner](http://www.kamalnigam.com/papers/emactive-conald98.pdf) as well as a random learner using SVM as a multiclass classifier. For the active learner, we select points of uncertainity by choosing distances whose difference from the decision boundary is minimum. This is made possible by using the *'decision_fuction'* from *'sklearn's* SVM classifier. The random learner chooses points at random for the classification task, whereas the active learner '*actively*' chooses point to be queried, as discussed below.

#### Uncertainity sampling
Consider the below point P1 (Point 1) at a distance of d11 and d12 from the hyperplane 1 and hyperplane2 respectively, say H1 and H2. Similarly, P2 is at a distance of d21 and d22 from the the hyperplanes H1 and H2 as well. Our uncertainity sampling method would choose P1 because the absolute difference of d11 and d12 is less than the aboslute difference of d21 an d22.
There are various sampling methods based on probabilities or entropy, but we chose to use this not only because it was much simpler to understand and implement, but also because it worked quite well in query selection.
![alt text](https://github.com/MascarenhasV/bioimageClassifier/blob/master/images/active_selection.PNG "Decision Boundaries")


We start with an initial batch size of randomly selected points for both the classifiers, and vary the batch sizes and the number of total available calls to the oracle. We did not perform a grid search of the parameters for the classifer, but it could be done as future work.

We experimented with SVM as the base learner with linear kernel, but it did not give us good results. We moved to SVM with RBF kernel, and observed a great improvement in the results. Other base learners like Decision Trees, Random Forests, Naive Bayes were also used, but they did not give us good results. We also used a multilayer perceptron neural network as well, but the examples were too less for the NN to converge. SVM with RBF as kernel worked well for us and could have been improved since we used the default C and gamma values. We leave the parameter tuning part by grid search over the parameter space for future work.

We also used '*SelectKBest*' from *'sklearn'* to perform feature selection, and trimmed down the desired number of features to k=26 on the difficult data set, which had irrelevant features. 

### Results:
The following results were observed with fixed random seeds so that we can replicate the results. Even if the random seed is not initialized, the results were pretty much similar.

#### Predictions for the active learner :
The below results were obtained for the difficult data set which had a lot of noise and half the features were noisy as well. For the other datasets, the result improved since they had less noise, and can be seen in the plots below.

```
Classification report for classifier OneVsRestClassifier(estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False),
          n_jobs=1):
                       precision    recall  f1-score   support

                Actin       0.83      0.86      0.85       128
Endoplasmic_Reticulum       0.89      0.89      0.89       112
            Endosomes       0.97      1.00      0.99       116
             Lysosome       0.96      0.99      0.98       132
         Microtubules       0.76      0.81      0.78       129
         Mitochondria       0.76      0.75      0.76       139
          Peroxisomes       0.84      0.85      0.85       131
      Plasma_Membrane       0.88      0.74      0.80       113

          avg / total       0.86      0.86      0.86      1000

```
The confusion matrix is given as follows:
```
Confusion matrix:
[[110   1   0   0  14   1   0   2]
 [  0 100   0   3   0   8   0   1]
 [  0   0 116   0   0   0   0   0]
 [  0   0   0 131   0   0   1   0]
 [  7   1   0   0 104  12   5   0]
 [  2   6   0   2  16 104   7   2]
 [  1   2   3   0   1   5 112   7]
 [ 12   2   0   0   1   6   8  84]]

 ```
#### Predictions for the random learner :
 ```
Classification report for classifier OneVsRestClassifier(estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=0, shrinking=True,
  tol=0.001, verbose=False),
          n_jobs=1):
                       precision    recall  f1-score   support

                Actin       0.81      0.84      0.83       128
Endoplasmic_Reticulum       0.82      0.92      0.87       112
            Endosomes       0.96      1.00      0.98       116
             Lysosome       0.95      0.95      0.95       132
         Microtubules       0.73      0.76      0.74       129
         Mitochondria       0.76      0.71      0.74       139
          Peroxisomes       0.82      0.82      0.82       131
      Plasma_Membrane       0.90      0.73      0.81       113

          avg / total       0.84      0.84      0.84      1000
```
The confusion matrix is given as follows:
```
Confusion matrix:
Confusion matrix:
[[108   4   0   0  14   1   0   1]
 [  0 103   0   3   0   6   0   0]
 [  0   0 116   0   0   0   0   0]
 [  0   1   1 126   0   1   3   0]
 [ 11   2   0   0  98  12   6   0]
 [  1   8   0   3  20  99   7   1]
 [  2   3   4   0   2   6 107   7]
 [ 11   4   0   1   1   5   8  83]]
```

As we see in the plot below, the active learner clearly outperforms the random learner. We see the best results when the active learner used much less queried labels than the random learner to reach the higher accuracy (or lower test error). 

#### Easy dataset:
![alt text](https://github.com/MascarenhasV/bioimageClassifier/blob/master/images/easy_batch_50.png "Easy dataset with batch size 50")
#### Moderate dataset:
![alt text](https://github.com/MascarenhasV/bioimageClassifier/blob/master/images/moderate_batch_50.png "Moderate dataset with batch size 50")
#### Hard dataset:
![alt text](https://github.com/MascarenhasV/bioimageClassifier/blob/master/images/difficult_batch_50.png "Difficult dataset with batch size 50")

In almost all cases (without random seeds), the active learner did better than the random learner. We also keep track of the best model in case the active learner overfits after achieving the best results. We used batch size as 50 so that the curves shown above are smoother. A lower batch size would give similar accuracy in the end, but observe a zagged set of curves. A higher batch size would not make use of active learner's strategy to choose points which are the best, and hence would perform similar to a random learner. 


#### People Involved: 
- Mick Zomnir
- Vipul Mascarenhas

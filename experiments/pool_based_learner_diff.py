import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import metrics
import random
import heapq
from sklearn.feature_selection import RFE

# load the training data
dataframe_train = pd.read_csv('../Data/DIFFICULT_TRAIN.csv', header=None)
dataset_train = dataframe_train.values
# choose the first 26 columns as training examples
data_train = dataframe_train.ix[:,0:51]
# choose the 27th column as the training labels
labels_train = dataframe_train.ix[:,52]

print data_train.shape
print labels_train.shape

# initialize size of labelled pool of data
labeled_pool_size = 100
calls_to_oracle = 2500

cost = labeled_pool_size
#calls_to_oracle = calls_to_oracle - labeled_pool_size


# randomly sample data into labeled and unlabeled instances
train_labeled_indices = random.sample(range(0,len(data_train)), labeled_pool_size)

# create mask for the indices in the training data
mask = data_train.index.isin(train_labeled_indices)

# split the training data in labeled and unlabeled data using the mask
labeled_instances = np.array(data_train[mask])
labeled_instances_labels = np.array(labels_train[mask])
unlabeled_instances =  np.array(data_train[~mask])
unlabeled_instances_labels = np.array(labels_train[~mask])


r_labeled_instances = np.array(data_train[mask])
r_labeled_instances_labels = np.array(labels_train[mask])
r_unlabeled_instances =  np.array(data_train[~mask])
r_unlabeled_instances_labels = np.array(labels_train[~mask])


label_dict = {'0':'Actin', '1':'Endoplasmic_Reticulum', '2':'Endosomes', '3':'Lysosome','4':'Microtubules','5':'Mitochondria',
              '6':'Peroxisomes', '7':'Plasma_Membrane'}
batch_size = 50

clf = None
r_clf = None

while cost < calls_to_oracle:

    #clf = OneVsRestClassifier(SVC(random_state=0))

    clf = SVC(random_state=0, decision_function_shape='ovr')

    selector = RFE(SVC(random_state=0, decision_function_shape='ovr', kernel="linear"), n_features_to_select=26 , step=1)

    selector.fit(labeled_instances, labeled_instances_labels)
    #new_feature_indices = [i for i, x in enumerate(selector.support_) if x ]

    new_feature_indices = selector.get_support(indices=True)
    print new_feature_indices

    new_features = labeled_instances[:,new_feature_indices]
    clf.fit( new_features, labeled_instances_labels)

    #labels_predicted = clf.predict(unlabeled_instances)

    ## TO DO : distance function resuts, write logic here -> query_indices
    labels_prob = clf.decision_function(unlabeled_instances[:,new_feature_indices])
    query_indices = []
    query_indices_labels = []

    data_heap = []

    for index in range(0, len(labels_prob)):
        current_row = np.array(labels_prob[index])
        inferred_label = label_dict[str(np.argmax(current_row))]
        top_two = np.sort(current_row)[::-1][0:2]
        difference = top_two[0] - top_two[1]
        heapq.heappush(data_heap, (np.abs(difference), index, inferred_label))

    min_gap = heapq.nsmallest(batch_size, data_heap)

    # remind Mick to explain why this was named min_gap and also, complete this implementation later
    #max_gap = heapq.nlargest(batch_size,data_heap)

    for row in min_gap:
        query_indices.append(row[1])
        #query_indices_labels.append(row[2])

    # write logic to extract that point from the unlabeled_instances
    inferred_data = unlabeled_instances[query_indices]
    inferred_data_labels = unlabeled_instances_labels[query_indices]  # queried labels

    cost = cost + len(query_indices)

    # print "Labeled :Before"
    # print labeled_instances.shape
    # print labeled_instances_labels.shape

    labeled_instances = np.vstack((labeled_instances,inferred_data))
    labeled_instances_labels = np.concatenate((labeled_instances_labels,inferred_data_labels))

    unlabeled_instances = np.delete(unlabeled_instances, query_indices, 0)
    unlabeled_instances_labels = np.delete(unlabeled_instances_labels, query_indices, None)
    print "Cost: " + str(cost)


    r_clf = OneVsRestClassifier(SVC(random_state=0))
    r_clf.fit(r_labeled_instances[:,new_feature_indices], r_labeled_instances_labels)
    r_labels_predicted = r_clf.predict(r_unlabeled_instances[:,new_feature_indices])

    random_indices = random.sample(range(0,r_unlabeled_instances.shape[0]), batch_size)
    random_instances = r_unlabeled_instances[random_indices]
    random_instance_labels = r_unlabeled_instances_labels[random_indices]

    r_labeled_instances = np.vstack((r_labeled_instances, random_instances))
    r_labeled_instances_labels = np.concatenate((r_labeled_instances_labels, random_instance_labels))

    r_unlabeled_instances = np.delete(r_unlabeled_instances, random_indices, 0)
    r_unlabeled_instances_labels = np.delete(r_unlabeled_instances_labels, random_indices, None)


    # if cost == 500:
    #     break


dataframe_test = pd.read_csv('../Data/DIFFICULT_TEST.csv', header=None)
dataset_test = dataframe_test.values
data_test = dataset_test[:,0:52].astype(float)
labels_expected = dataset_test[:,52]

labels_predicted = clf.predict(data_test[:,new_feature_indices])
r_labels_predicted = r_clf.predict(data_test[:,new_feature_indices])

print "Labeled Data size: " + str(labeled_instances.shape)

print("Classification report for classifier %s:\n%s\n"
  % (clf, metrics.classification_report(labels_expected, labels_predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, labels_predicted))


print("Classification report for random classifier %s:\n%s\n"
  % (clf, metrics.classification_report(labels_expected, r_labels_predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, r_labels_predicted))






import numpy
import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def main():
	dataframe_train = pandas.read_csv('Data\EASY_TRAIN.csv', header=None)
	dataset_train = dataframe_train.values
	data_train = dataset_train[:,0:26].astype(float)
	labels_train = dataset_train[:,26]
	clf = RandomForestClassifier(n_jobs=2)
	clf.fit(data_train,labels_train)

	dataframe_test = pandas.read_csv('Data\EASY_TEST.csv', header=None)
	dataset_test = dataframe_test.values
	data_test = dataset_test[:,0:26].astype(float)
	labels_expected = dataset_test[:,26]
	labels_predicted = clf.predict(data_test)

	print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(labels_expected, labels_predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels_expected, labels_predicted))


if __name__ == "__main__":
	main()
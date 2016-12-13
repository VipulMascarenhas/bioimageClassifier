from scipy.cluster.hierarchy import ward, dendrogram
from sklearn import cluster
import itertools
import pandas
import matplotlib.pyplot as plt
import numpy as np


dataframe_train = pandas.read_csv('Data\EASY_TRAIN.csv', header=None)
dataset_train = dataframe_train.values
data_train = dataset_train[:,0:26].astype(float)
labels_train = dataset_train[:,26]

arr = [None]*4120
for index in range(0, 4120):
	arr[index] = index + 1
titles = np.array(labels_train[1:1000])

print titles[0]

w = cluster.ward_tree(data_train[1:1000,:], return_distance=True)


linkage_matrix = ward(w[0]) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(64, 100)) # set size
ax = dendrogram(linkage_matrix, orientation="left", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters


import time
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import cluster, datasets, manifold
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

np.random.seed(0)


dataframe_train = pandas.read_csv('Data\EASY_TRAIN.csv', header=None)
dataset_train = dataframe_train.values
data_train = dataset_train[:,0:26].astype(float)
labels_train = dataset_train[:,26]



colors = np.array([x for x in 'bgrcmykwbgrcmykwbgrcmykwbgrcmykw'])
colors = np.hstack([colors] * 20)

clustering_names = [
    'MiniBatchKMeans']

plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1


datasets = [(data_train, labels_train)]

for i_dataset, dataset in enumerate(datasets):
    X, y = dataset
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # create clustering estimators
    two_means = cluster.MiniBatchKMeans(n_clusters=8)
    
    #affinity_propagation = cluster.AffinityPropagation(damping=.9,
    #                                                   preference=-200)

    #birch = cluster.Birch(n_clusters=8)

    clustering_algorithms = [
        two_means]

    for name, algorithm in zip(clustering_names, clustering_algorithms):
        # predict cluster memberships
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        print y_pred[0]

        # plot
        plt.subplot(4, len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        tsne1 = manifold.TSNE(n_components=2, init='pca', random_state=0)
        res1 = tsne1.fit_transform(X)
        plt.scatter(res1[:, 0], res1[:, 1], c=colors[y_pred].tolist(), cmap=plt.cm.Spectral)
        #plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(algorithm, 'cluster_centers_'):
            centers = algorithm.cluster_centers_
            center_colors = colors[:len(centers)]
            #plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            res = tsne.fit_transform(centers)
            plt.scatter(res[:, 0], res[:, 1], c=center_colors, cmap=plt.cm.Spectral, s=100)

        #plt.xlim(-2, 2)
        #plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1








plt.show()
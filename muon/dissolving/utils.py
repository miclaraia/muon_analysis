import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_curve

from keras.utils import np_utils

from dec_keras.DEC import DEC, ClusteringLayer, cluster_acc


def one_percent_fpr(y, pred, fom):
    fpr, tpr, thresholds = roc_curve(y, pred)
    FoM = 1-tpr[np.where(fpr<=fom)[0][-1]] # MDR at 1% FPR
    threshold = thresholds[np.where(fpr<=fom)[0][-1]]
    return FoM, threshold, fpr, tpr


def calc_f1_score(y_true, predicted_clusters, cluster_to_label_mapping):
    y_pred = []
    for i in range(len(y_true)):
          y_pred.append(cluster_to_label_mapping[predicted_clusters[i]])
    return f1_score(y_true, np.array(y_pred))

def get_cluster_to_label_mapping_safe(y, y_pred, n_classes, n_clusters, toprint=True):
    """Enusre at least one cluster assigned to each label.
    """
    one_hot_encoded = np_utils.to_categorical(y, n_classes)

    cluster_to_label_mapping = []
    n_assigned_list = []
    majority_class_fractions = []
    majority_class_pred = np.zeros(y.shape)
    for cluster in range(n_clusters):
        cluster_indices = np.where(y_pred == cluster)[0]
        n_assigned_examples = cluster_indices.shape[0]
        n_assigned_list.append(n_assigned_examples)
        cluster_labels = one_hot_encoded[cluster_indices]
        cluster_label_fractions = np.mean(cluster_labels, axis=0)
        majority_cluster_class = np.argmax(cluster_label_fractions)
        cluster_to_label_mapping.append(majority_cluster_class)
        majority_class_pred[cluster_indices] += majority_cluster_class
        majority_class_fractions.append(cluster_label_fractions[majority_cluster_class])
        if toprint:
            print(cluster, n_assigned_examples, majority_cluster_class, cluster_label_fractions[majority_cluster_class])
    #print(cluster_to_label_mapping)
    if toprint:
        print(np.unique(y), np.unique(cluster_to_label_mapping))
    try:
        # make sure there is at least 1 cluster representing each class
        assert np.all(np.unique(y) == np.unique(cluster_to_label_mapping))
    except AssertionError:
        # if there is no cluster for a class then we will assign a cluster to that
        # class

        # find which class it is
        # ASSUMPTION - this task is binary

        diff = list(set(np.unique(y)) - set(np.unique(cluster_to_label_mapping)))[0]
          # we choose the cluster that contains the most examples of the class with no cluster

        one_hot = np_utils.to_categorical(y_pred[np.where(y==diff)[0]], \
                                            len(cluster_to_label_mapping))

        cluster_to_label_mapping[np.argmax(np.sum(one_hot, axis=0))] = int(diff)
    if toprint:
        print(cluster_to_label_mapping)
    return cluster_to_label_mapping, n_assigned_list, majority_class_fractions


def pca_plot(base_network, x, cluster_centres, y=None, labels=[],
             lcolours=[], ulcolour='#747777', ccolour='#4D6CFA'):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(np.nan_to_num(base_network.predict(x)))
    c_pca = pca.transform(cluster_centres)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    if np.any(y):
        unique_targets = list(np.unique(y))
        if -1 in unique_targets:
            ax.scatter(
                x_pca[np.where(y == -1), 0],
                x_pca[np.where(y == -1), 1],
                marker='o',
                s=20,
                color=ulcolour, alpha=0.1)
            unique_targets.remove(-1)
        for l in unique_targets:
            l = int(l)
            ax.scatter(
                x_pca[np.where(y == l), 0],
                x_pca[np.where(y == l), 1],
                marker='o', s=5,
                color=lcolours[l],
                alpha=0.7,
                label=labels[l])
    else:
        ax.scatter(
            x_pca[:,0], x_pca[:,1],
            marker='o', s=20, color=ulcolour, alpha=0.7)
    ax.scatter(
        c_pca[:,0], c_pca[:,1],
        marker='o', s=40, color=ccolour, alpha=1.0, label='cluster centre')

    for i in range(len(cluster_centres)):
        ax.text(c_pca[i,0], c_pca[i,1], str(i), size=20)
    plt.axis('off')
    #plt.legend(ncol=2)
    plt.show()

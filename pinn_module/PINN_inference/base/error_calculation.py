import numpy as np


def Poisson_error(results, labels):
    delta = results - labels
    return np.sqrt(np.sum(np.square(delta))) / np.sqrt(np.sum(np.square(labels)))


def Schrodinger_error(results, labels):
    h_results = np.sqrt(np.square(results[:, 0]) + np.square(results[:, 1]))
    h_labels = np.sqrt(np.square(labels[:, 0]) + np.square(labels[:, 1]))
    h_error = h_labels - h_results
    return np.sqrt(np.sum(np.square(h_error))) / np.sqrt(np.sum(np.square(h_labels)))


def NS_error(results, labels):
    delta = results - labels
    L2_u = np.sqrt(np.sum(np.square(delta[:, 0]))) / np.sqrt(np.sum(np.square(labels[:, 0])))
    L2_v = np.sqrt(np.sum(np.square(delta[:, 1]))) / np.sqrt(np.sum(np.square(labels[:, 1])))
    return L2_u, L2_v

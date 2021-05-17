from collections import Counter
from time import sleep

from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import plot_partial_dependence
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph, KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

from clust.bayes import bayes
from clust.pdp_ice import pdp


def run():
    points = []
    courses = []
    with open('data.txt', 'r') as f:
        for line in f:
            num, ege, eng, language, course = line.split(' ')
            points.append((int(ege), int(eng), 0 if language == 'java' else 1))
            courses.append(int(course))
    X = np.array(points)
    y = np.array(courses)

    neigh = KNeighborsClassifier(n_neighbors=10)
    # neigh = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                       hidden_layer_sizes=(15,), random_state=1)
    kernel = 1.0 * RBF(1.0)
    # neigh = GaussianProcessClassifier(kernel=kernel,
    #      random_state=0)
    neigh.fit(X, y)

    print(neigh.predict([[200, 86, 1]]))

    kneighbors_graph(X, 19, 'connectivity')

    features = [(0, 1)]
    pdp(neigh, X, features)

    bayes(neigh, X)


if __name__ == '__main__':
    run()

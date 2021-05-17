import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, kneighbors_graph, KNeighborsClassifier
from matplotlib import pyplot as plt

from clust.pdp_ice import pdp


def bayes(model, X):

    y = model.predict(X)

    gnb = GaussianNB()
    gnb.fit(X, y)

    print(gnb.predict([[200, 86, 1]]))


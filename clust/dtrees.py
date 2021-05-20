from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def dtree(model, X):
    clf = DecisionTreeClassifier(random_state=0, max_depth=3)
    y = model.predict(X)
    clf.fit(X, y)
    cross_val_score(clf, X, y, cv=10)

    px = 1 / plt.rcParams['figure.dpi']
    plt.subplots(figsize=(1920 * px, 1080 * px))
    tree.plot_tree(clf)
    plt.show()

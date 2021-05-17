from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


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

    # dtree = tree.DecisionTreeClassifier(random_state=0)
    # dtree.fit(X, y)
    #
    # print(dtree.predict([[200, 86, 1]]))
    #
    # tree.plot_tree(dtree, filled=True)

    plot_step = 0.02
    n_classes = 5
    plot_colors = "ryb"
    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = X[:, pair]
        y = y

        # Train
        clf = DecisionTreeClassifier().fit(X, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        # plt.xlabel(iris.feature_names[pair[0]])
        # plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")

    plt.figure()
    clf = DecisionTreeClassifier().fit(X, y)
    plot_tree(clf, filled=True)
    plt.show()


if __name__ == '__main__':
    run()

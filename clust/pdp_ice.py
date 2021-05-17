from sklearn.inspection import plot_partial_dependence
from matplotlib import pyplot as plt


def pdp(model, X, features):
    # plt.figure(figsize=(10, 9))
    # fig = plt.gcf()
    plot_partial_dependence(model, X, features, target=4)
    # plot_partial_dependence(model, X, features, kind='both', target=1)
    # plot_partial_dependence(model, X, features, kind='both', target=2)
    # plot_partial_dependence(model, X, features, kind='both', target=3)
    # plot_partial_dependence(model, X, features, kind='both', target=4)
    # plot_partial_dependence(model, X, features, kind='both', target=5)
    # fig.savefig('test2png.png', dpi=100)
    plt.show()
    # plt.gca()
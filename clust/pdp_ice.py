from sklearn.inspection import plot_partial_dependence
from matplotlib import pyplot as plt


def pdp(model, X, features):
    # plt.figure(figsize=(10, 9))
    # fig = plt.gcf()
    # plot_partial_dependence(model, X, features, target=4)
    fig1 = plot_partial_dependence(model, X, features, kind='average', target=1)
    fig2 = plot_partial_dependence(model, X, features, kind='average', target=2)
    fig3 = plot_partial_dependence(model, X, features, kind='average', target=3)
    fig4 = plot_partial_dependence(model, X, features, kind='average', target=4)
    fig5 = plot_partial_dependence(model, X, features, kind='average', target=5)
    # fig.savefig('test2png.png', dpi=100)
    plt.show()
    # plt.gca()

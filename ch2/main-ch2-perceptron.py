import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from plot_functions import plot_decision_regions


if __name__ == '__main__':
    df = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/iris/iris.data', header=None)
    df.tail()
    
    # select setosa and versicolor
    targets = df.iloc[0:100, 4].values
    targets = np.where(targets == 'Iris-setosa', -1, 1)
    
    # extract sepal length and petal length
    training_features = df.iloc[0:100, [0, 2]].values
    
    # plot data
    plt.scatter(training_features[:50, 0], training_features[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(training_features[50:100, 0], training_features[50:100, 1],
                color='blue', marker='x', label='versicolor')
    
    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('./iris_1.png', dpi=300)
    plt.show()
    ppn = Perceptron(eta=0.1, n_iter=10)
    
    ppn.fit(training_features, targets)
    
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    
    plt.tight_layout()
    # plt.savefig('./perceptron_1.png', dpi=300)
    plt.show()
    
    plot_decision_regions(training_features, targets, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()

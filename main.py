from mlp import MLP
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    training = {}
    test = {}

    training['labels'] = np.genfromtxt('iris_labels_training.csv', delimiter=',')
    test['labels'] = np.genfromtxt('iris_labels_test.csv', delimiter=',')
    training['features'] = np.genfromtxt('iris_samples_training.csv', delimiter=',')
    test['features'] = np.genfromtxt('iris_samples_test.csv', delimiter=',')

    network = MLP((4, 7, 3))
    network.fit(training['features'], training['labels'], epochs=5000, stop_criteria=0.0001, default_window=1000)
    network.predict(test['features'], test['labels'])

    # Display Confusion Matrix
    print('')
    print('\tConfusion Matrix')
    for line in network.confusion_matrix:
        print('\t{}'.format(line))

    # Display performance measurements
    print('')
    print('\tOverall Accuracy \t{:0.2f}%'.format(network.performance['accuracy']*100))
    print('\tOverall Precision\t{:0.2f}%'.format(network.performance['precision']*100))
    print('\tOverall Sensitivity \t{:0.2f}%'.format(network.performance['sensitivity']*100))
    print('\tOverall Specificity \t{:0.2f}%'.format(network.performance['specificity']*100))

    # Display Weights
    print('')
    for index in range(len(network.weights)):
        print('')
        print('\tPesos entre Camadas {} e {}'.format(index, index + 1))
        for line in network.weights[index]:
            print('\t\t{}'.format(line))

    # Plot moving average
    plt.plot(network.moving_average)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()

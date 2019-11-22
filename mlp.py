""" 
|**********************************************************************|
*
* Project            Multilayer Perceptron Implementation
*
* Author(s)          Thiago Martin, Matheus Borges e Rodrigo Dallagnol
*
* Date created       22/11/2019
*
* Purpose            Proof of concept
*
* Current Version    v1.0
*
* Copyright (c) 2019 Thiago Martin
* This code is licensed under MIT license (see LICENSE for details)
*
* Feel free to use this code however you wish.
* Do not take knowledge for granted, spread it!
*
|**********************************************************************| 
"""

import random
import numpy as np

TP = 0
FN = 1
FP = 2
TN = 3

np.set_printoptions(precision=3)


class MLP:

    def __init__(self, shape, activation='sigmoid'):
        self.layers = []
        self.shape = shape
        self.activation = activation
        self.n_classes = shape[-1]
        self.weights = []
        n = len(shape)

        for i in range(n):
            self.layers.append(np.ones(self.shape[i]))

        for i in range(n-1):
            self.weights.append(np.zeros((self.shape[i], self.shape[i + 1])))

        self.previous_weights_variation = [0, ] * len(self.weights)
        self.initializeWeights()

    def actv_func(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return

    def actv_func_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'tanh':
            return 1.0-x**2
        else:
            return

    def initializeWeights(self):
        for i in range(len(self.weights)):
            self.weights[i][...] = np.random.random(
                (self.shape[i], self.shape[i+1])) - 0.5

    def resetWeights(self):
        self.initializeWeights()

    def feedForward(self, X):
        self.layers[0][...] = X

        for i in range(1, len(self.shape)):
            x = np.dot(self.layers[i-1], self.weights[i-1])
            self.layers[i][...] = self.actv_func(x)

        return self.layers[-1]

    def backPropagation(self, Y, learning_rate, momentum):
        deltas = []
        error = Y - self.layers[-1]
        error_sq_sum = (error**2).sum()

        derivative_values = self.actv_func_derivative(self.layers[-1])
        output_layer_deltas = error * derivative_values
        deltas.append(output_layer_deltas)

        n_hidden_layers = len(self.shape) - 2
        for i in range(n_hidden_layers, 0, -1):
            derivative_values = self.actv_func_derivative(self.layers[i])
            current_layer_deltas = np.dot(
                deltas[0], self.weights[i].T) * derivative_values
            deltas.insert(0, current_layer_deltas)

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            weights_variation = np.dot(layer.T, delta)

            self.weights[i] += learning_rate * weights_variation + \
                momentum * self.previous_weights_variation[i]
            self.previous_weights_variation[i] = weights_variation

        return error_sq_sum

    def fit(self, X, Y, epochs=30000, learning_rate=.1, momentum=0.1, stop_criteria=None, default_window=100):
        self.errors = []
        self.moving_average = []

        for i in range(epochs):
            n = np.random.randint(len(X))
            self.feedForward(X[n])
            error = self.backPropagation(Y[n], learning_rate, momentum)
            self.errors.append(error)

            if stop_criteria:
                window_size = min(i + 1, default_window)
                current_average = sum(self.errors[-window_size:]) / window_size
                self.moving_average.append(current_average)
                if current_average < stop_criteria:
                    break

        return self.errors

    def predict(self, X, Y=None):
        self.predictions = np.zeros((len(X), self.n_classes))

        for index in range(len(X)):
            self.predictions[index] = self.feedForward(X[index])

        if Y is not None:
            self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
            predicted_labels = []

            for p in self.predictions:
                predicted_labels.append(np.where(p == np.max(p))[0])

            expected_labels = np.where(Y == np.max(Y))[1]

            for predicted, expected in zip(predicted_labels, expected_labels):
                self.confusion_matrix[expected][predicted] += 1

            self.performanceMeasures(self.confusion_matrix)
        return self.predictions

    def safeDivision(self, n, d):
        return n / d if d else 0

    def performanceMeasures(self, matrix):
        class_info = np.zeros((self.n_classes, 4))
        precisions = np.zeros((self.n_classes))
        sensitivities = np.zeros((self.n_classes))
        specificities = np.zeros((self.n_classes))
        accuracy = 0

        n_samples = matrix.sum()

        for n in range(self.n_classes):
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    if i == n and j == n:
                        class_info[n][TP] = matrix[i][j]
                    elif i == n and j != n:
                        class_info[n][FN] += matrix[i][j]
                    elif i != n and j == n:
                        class_info[n][FP] += matrix[i][j]
                    else:
                        class_info[n][TN] += matrix[i][j]

        for n in range(self.n_classes):
            accuracy += class_info[n][TP]
            precisions[n] += self.safeDivision(class_info[n]
                                               [TP], (class_info[n][TP] + class_info[n][FP]))
            sensitivities[n] += self.safeDivision(
                class_info[n][TP], (class_info[n][TP] + class_info[n][FN]))
            specificities[n] += self.safeDivision(
                class_info[n][TN], (class_info[n][TN] + class_info[n][FP]))

        accuracy /= n_samples
        precision = precisions.sum() / self.n_classes
        sensitivity = sensitivities.sum() / self.n_classes
        specificity = specificities.sum() / self.n_classes

        self.performance = {}
        self.performance['tp'] = class_info[:, TP]
        self.performance['fn'] = class_info[:, FN]
        self.performance['fp'] = class_info[:, FP]
        self.performance['tn'] = class_info[:, TN]

        self.performance['accuracy'] = accuracy
        self.performance['precision'] = precision
        self.performance['sensitivity'] = sensitivity
        self.performance['specificity'] = specificity

        self.performance['precisions'] = precisions
        self.performance['sensitivities'] = sensitivities
        self.performance['specificities'] = specificities

        return self.performance

import numpy as np

def ReLU(array):
    """Applies the ReLU activation function to the input array."""
    return np.maximum(0, array)

def softmax(array):
    """Applies the softmax activation function to the input array."""
    array = array - np.max(array)
    exp_array = np.exp(array)
    return exp_array / np.sum(exp_array)

def neural_network_neurons(W, B, X):
    """Calculates the output of a neural network layer given weights, biases, and input."""
    return np.matmul(W, X) + B
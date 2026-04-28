import torch

def ReLU(tensor):
    """Applies the ReLU function element-wise to the input tensor."""
    return torch.maximum(torch.tensor(0.0), tensor)

def softmax(tensor):
    """Applies the softmax function to the input tensor."""
    exp_tensor = torch.exp(tensor - torch.max(tensor))
    return exp_tensor / torch.sum(exp_tensor)

def neural_network_neurons(W, B, X):
    """Calculates the output of a neural network layer given weights, biases, and input."""
    return torch.matmul(W, X) + B
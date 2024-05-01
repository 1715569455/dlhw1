import numpy as np
import os
import gzip
from copy import deepcopy

class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = (self.input > 0) * grad_output
        return grad_input


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return self.output * (1 - self.output) * grad_output


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return (1 - self.output**2) * grad_output


def create_activation_function(name):
    if name == 'relu':
        return ReLU()
    elif name == 'sigmoid':
        return Sigmoid()
    elif name == 'tanh':
        return Tanh()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class MLP:
    def __init__(self, dim_in, dim_hidden, dim_out, activ_name='relu'):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.weight1 = np.random.randn(self.dim_in, self.dim_hidden) * 0.01
        self.bias1 = np.zeros((1, self.dim_hidden))
        self.weight2 = np.random.randn(self.dim_hidden, self.dim_hidden) * 0.01
        self.bias2 = np.zeros((1, self.dim_hidden))
        self.weight3 = np.random.randn(self.dim_hidden, self.dim_out) * 0.01
        self.bias3 = np.zeros((1, self.dim_out))

        self.activ_layer1 = create_activation_function(activ_name)
        self.activ_layer2 = create_activation_function(activ_name)

        self.out1 = None
        self.out2 = None
        self.out3 = None
        self.act1 = None
        self.act2 = None
        self.act3 = None

        self.inputs = None
        self.bias1_grad = None
        self.bias2_grad = None
        self.bias3_grad = None
        self.weight1_grad = None
        self.weight2_grad = None
        self.weight3_grad = None

    def forward(self, inputs):  # inputs shape (B, dim_in)
        self.inputs = inputs
        self.out1 = np.dot(inputs, self.weight1) + self.bias1
        self.act1 = self.activ_layer1.forward(self.out1)
        self.out2 = np.dot(self.out1, self.weight2) + self.bias2
        self.act2 = self.activ_layer2.forward(self.out2)
        self.out3 = np.dot(self.out2, self.weight3) + self.bias3

        return self.out3

    def backward(self, grad):
        b = len(grad)
        self.bias3_grad = np.sum(grad, axis=0) / b
        self.weight3_grad = np.dot(self.out2.T, grad) / b
        grad = np.dot(grad, self.weight3.T)
        grad = self.activ_layer2.backward(grad)
        self.bias2_grad = np.sum(grad, axis=0) / b
        self.weight2_grad = np.dot(self.out1.T, grad) / b
        grad = np.dot(grad, self.weight2.T)
        grad = self.activ_layer1.backward(grad)
        self.bias1_grad = np.sum(grad, axis=0) / b
        self.weight1_grad = np.dot(self.inputs.T, grad) / b

    def update(self, lr, weight_decay):
        self.weight1 = self.weight1 - lr * (self.weight1_grad + weight_decay * self.weight1 ** 2)
        self.weight2 = self.weight2 - lr * (self.weight2_grad + weight_decay * self.weight2 ** 2)
        self.weight3 = self.weight3 - lr * (self.weight3_grad + weight_decay * self.weight3 ** 2)
        self.bias1 = self.bias1 - lr * (self.bias1_grad + weight_decay * self.bias1 ** 2)
        self.bias2 = self.bias2 - lr * (self.bias2_grad + weight_decay * self.bias2 ** 2)
        self.bias3 = self.bias3 - lr * (self.bias3_grad + weight_decay * self.bias3 ** 2)

    def load(self, param_file):
        parameters = np.load(param_file)
        self.weight1 = parameters['weight1']
        self.weight2 = parameters['weight2']
        self.weight3 = parameters['weight3']
        self.bias1 = parameters['bias1']
        self.bias2 = parameters['bias2']
        self.bias3 = parameters['bias3']

    def save(self, save_dir):
        np.savez(os.path.join(save_dir, 'parameters.npz'), weight1=self.weight1, bias1=self.bias1, weight2=self.weight2
                 , bias2=self.bias2, weight3=self.weight3, bias3=self.bias3)


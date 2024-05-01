import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    one_hot_labels = np.eye(10)[labels]
    return images, one_hot_labels


class DataLoader:
    def __init__(self, X, y, batch_size=1024):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = X.shape[0]
        self.indices = np.arange(self.num_samples)
        np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        else:
            batch_indices = self.indices[self.current_index:self.current_index+self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = self.y[batch_indices]
            self.current_index += self.batch_size
            return X_batch, y_batch


class SoftmaxCrossEntropy:
    def __init__(self):
        self.softmax_output = None
        self.targets = None

    def forward(self, outputs, targets):
        self.targets = targets
        self.epsilon = 1e-4
        exp_values = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.softmax_output = probabilities
        loss = -np.sum(targets * np.log(probabilities + self.epsilon)) / len(outputs)
        return loss

    def backward(self):
        return (self.softmax_output - self.targets)


class SGD:
    def __init__(self, model, loss, lr, weight_decay):
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay


    def step(self):
        grad_in = self.loss.backward()
        self.model.backward(grad_in)
        self.model.update(self.lr, self.weight_decay)


def lr_schedule(epoch_num, max_lr, min_lr, period):
    if epoch_num < period:
        return max_lr - (max_lr-min_lr) * epoch_num / period
    else:
        return min_lr

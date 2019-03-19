import tensorflow as tf
import numpy as np

def load_mnist(n_data=None):
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

    # randomize order
    permutation = np.random.permutation(len(train_labels))
    train_data = train_data[permutation]
    train_labels = train_labels[permutation]
    permutation = np.random.permutation(len(test_labels))
    test_data = test_data[permutation]
    test_labels = test_labels[permutation]

    # normalize, reshape, and convert to one-hot vectors
    train_data = np.reshape(train_data, (-1, 784)) / (255./2.) - 1.
    test_data = np.reshape(test_data, (-1, 784)) / (255./2.) - 1.
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    if n_data is not None:
        data = {'train_data': train_data[:n_data], 'train_labels': train_labels[:n_data], 'test_data': test_data[:n_data], 'test_labels': test_labels[:n_data]}
    else:
        data = {'train_data': train_data, 'train_labels': train_labels, 'test_data': test_data, 'test_labels': test_labels}

    return data


def one_hot(x, n_classes=None):
    # input: 1D array of N labels, output: N x max(x)+1 array of one-hot vectors
    if n_classes is None:
        n_classes = max(x) + 1

    x_one_hot = np.zeros([len(x), n_classes])
    x_one_hot[np.arange(len(x)), x] = 1
    return x_one_hot


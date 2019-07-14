import tensorflow as tf


def download_mnist():
    data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    return data[0], data[1]


x_train, y_train, x_test, y_test = download_mnist()

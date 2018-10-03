# -*- coding=utf-8 -*-
import numpy as np
import random
import gzip
import pickle


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


class Netwrk(object):
    """ neural network gradient descent """

    def __init__(self, sizes):

        self.sizes = len(sizes)  # 神经网络有几层
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y,1)for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self,training_data, epochs, mini_batch_size, eta, test_data=None):
        """ stochastic gradient descent  
        epochs: 训练多少轮
        mini_batch_size: 每一个sample包含多少个实例
        eta: 学习率
        """

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data) # 对训练集数据进行乱序(重新洗牌)
            mini_batchs = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 以mini_batch_size = 10 为例: 
            # mini_bachs = [[0,1,2...,9], [10, ..., 19], [20,...,29], ....]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete.".format(j))
        
    def update_mini_batch(self, mini_batch, eta):
        """ update the network's weight and biases 
        by applying gradient descent using backpropagation
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [x+y for x,y in zip(nabla_b, delta_nabla_b)]
            nabla_w = [x+y for x,y in zip(nabla_w, delta_nabla_w)]

        m = len(mini_batch)
        self.weights = [w1 - (eta/m)*w2 for w1 , w2 in zip(self.weights, nabla_w)]
        self.biases = [b1 - (eta/m)*b2 for b1, b2 in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

def mnist_data_load():
    with gzip.open('./neural-networks-and-deep-learning/data/mnist.pkl.gz', 'rb') as f:
        tr_d, va_d, te_d = pickle.load(f, encoding='bytes')
        # tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)



if __name__ == "__main__":
    net = Netwrk([784,30,10])
    training_data, validation_data, test_data = mnist_data_load()

    net.SGD(training_data=list(training_data), epochs=100, mini_batch_size=100, eta=0.5, test_data=list(test_data))


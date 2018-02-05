from mxnet import ndarray as nd
from mxnet import autograd
import random


class Config:
    def __init__(self):
        self.num_inputs = 3
        self.training_size = 1000
        self.batch_size = 10
        self.learning_rate = 1e-2
        self.num_epochs = 5


config = Config()


true_w = [2.5, 4.7, -3.2]
true_b = 2.9

X = nd.random_normal(shape=(config.training_size, config.num_inputs))
y = nd.dot(X, nd.array(true_w)) + true_b
y += 0.01 * nd.random_normal(shape=y.shape)


def data_generator(batch_size):
    index = list(range(config.training_size))
    random.shuffle(index)

    for i in range(0, config.training_size, batch_size):
        j = nd.array(index[i:min(i + batch_size, config.training_size)])
        yield nd.take(X, j), nd.take(y, j)


w = nd.random_normal(shape=(config.num_inputs, 1))
b = nd.zeros((1,))
parameters = [w, b]

for parameter in parameters:
    parameter.attach_grad()


def model(_input):
    return nd.dot(_input, w) + b


def square_loss(y_pred, y_true):
    return ((y_pred - y_true.reshape(y_pred.shape)) ** 2) / 2


def SGD(_parameters, learning_rate):
    for _parameter in _parameters:
        _parameter -= learning_rate * _parameter.grad


for e in range(config.num_epochs):
    total_loss = 0

    for data, label in data_generator(config.batch_size):
        with autograd.record():
            _output = model(data)
            loss = square_loss(_output, label)
        loss.backward()
        SGD(parameters, config.learning_rate)
        total_loss += nd.sum(loss).asscalar()

    print("epoch: [%d], training loss: [%.4f]" % (e+1, total_loss))

print("true parameters:", true_w, true_b)
print("trained parameters:", parameters)
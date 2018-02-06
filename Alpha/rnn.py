import mxnet as mx
from mxnet import ndarray as nd


class Config:
    def __init__(self):
        self.input_dim = 128
        self.hidden_dim = 256
        self.output_dim = 128
        self.num_steps = 5

        self.std = 1e-2

        self.learning_rate = 1e-2
        self.num_epochs = 5
        self.batch_size = 32


config = Config()


def get_parameters():
    W_xh = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hh = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_h = nd.zeros(config.hidden_dim)

    W_hy = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.output_dim))
    b_y = nd.zeros(config.output_dim)

    parameters = [W_xh, W_hh, b_h, W_hy, b_y]
    for parameter in parameters:
        parameter.attach_grad()

    return parameters


# RNN forward operation
def rnn(_inputs, initial_state, *parameters):
    # _inputs: a list with length num_steps,
    # corresponding element: batch_size * input_dim matrix
    H = initial_state
    W_xh, W_hh, b_h, W_hy, b_y = parameters
    _outputs = []

    for X in _inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        _outputs.append(Y)

    return _outputs, H


# gradient clipping to avoid gradient explosion
def gradient_clipping(parameters, threshold, ctx):
    if threshold is not None:
        norm = nd.array([0.0], ctx)

        for parameter in parameters:
            norm += nd.sum(parameter.grad ** 2)
        norm = nd.sqrt(norm).asscalar()

        if norm > threshold:
            for parameter in parameters:
                parameter.grad[:] *= (threshold / norm)


if __name__ == '__main__':
    initial_state = nd.zeros(shape=(config.batch_size, config.hidden_dim))
    dump_data = [nd.random_normal(shape=(config.batch_size, config.input_dim)) for _ in range(config.num_steps)]

    parameters = get_parameters()
    _outputs, final_state = rnn(dump_data, initial_state, *parameters)

    print(_outputs, final_state)

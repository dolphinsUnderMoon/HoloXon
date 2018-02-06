from mxnet import nd


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
    # parameters for UPDATE gate
    W_xz = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hz = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_z = nd.zeros(shape=config.hidden_dim)

    # parameters for RESET gate
    W_xr = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hr = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_r = nd.zeros(shape=config.hidden_dim)

    # parameters for candidate hidden state
    W_xh = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hh = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_h = nd.zeros(shape=config.hidden_dim)

    # output layer
    W_hy = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.output_dim))
    b_y = nd.zeros(shape=config.output_dim)

    parameters = [W_xz, W_hz, b_z,
                  W_xr, W_hr, b_r,
                  W_xh, W_hh, b_h,
                  W_hy, b_y]

    for parameter in parameters:
        parameter.attach_grad()

    return parameters


def gru(_inputs, initial_state, *parameters):
    # _inputs: a list with length num_steps,
    # corresponding element: batch_size * input_dim matrix

    H = initial_state

    [W_xz, W_hz, b_z,
     W_xr, W_hr, b_r,
     W_xh, W_hh, b_h,
     W_hy, b_y] = parameters

    _outputs = []

    for X in _inputs:
        # compute update gate from input and last/initial hidden state
        update_gate = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        # compute reset gate from input and last/initial hidden state
        reset_gate = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        # compute candidate hidden state from input, reset gate and last/initial hidden state
        H_candidate = nd.tanh(nd.dot(X, W_xh) + reset_gate * nd.dot(H, W_hh) + b_h)
        # compute hidden state from candidate hidden state and last hidden state
        H = update_gate * H + (1 - update_gate) * H_candidate
        # compute output from hidden state
        Y = nd.dot(H, W_hy) + b_y
        _outputs.append(Y)

    return _outputs, H


if __name__ == '__main__':
    initial_state = nd.zeros(shape=(config.batch_size, config.hidden_dim))
    dump_data = [nd.random_normal(shape=(config.batch_size, config.input_dim)) for _ in range(config.num_steps)]

    parameters = get_parameters()
    _outputs, final_state = gru(dump_data, initial_state, *parameters)

    print(_outputs, final_state)
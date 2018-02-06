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
    # parameters for INPUT gate
    W_xi = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hi = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_i = nd.zeros(shape=config.hidden_dim)

    # parameters for FORGET gate
    W_xf = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hf = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_f = nd.zeros(shape=config.hidden_dim)

    # parameters for OUTPUT gate
    W_xo = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_ho = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_o = nd.zeros(shape=config.hidden_dim)

    # parameters for memory cell
    W_xc = nd.random_normal(scale=config.std, shape=(config.input_dim, config.hidden_dim))
    W_hc = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.hidden_dim))
    b_c = nd.zeros(shape=config.hidden_dim)

    # output layer
    W_hy = nd.random_normal(scale=config.std, shape=(config.hidden_dim, config.output_dim))
    b_y = nd.zeros(shape=config.output_dim)

    parameters = [W_xi, W_hi, b_i,
                  W_xf, W_hf, b_f,
                  W_xo, W_ho, b_o,
                  W_xc, W_hc, b_c,
                  W_hy, b_y]

    for parameter in parameters:
        parameter.attach_grad()

    return parameters


def lstm(_inputs, initial_state_h, initial_state_c, *parameters):
    # _inputs: a list with length num_steps,
    # corresponding element: batch_size * input_dim matrix

    H = initial_state_h  # hidden state
    C = initial_state_c  # memory cell

    [W_xi, W_hi, b_i,
     W_xf, W_hf, b_f,
     W_xo, W_ho, b_o,
     W_xc, W_hc, b_c,
     W_hy, b_y] = parameters

    _outputs = []

    for X in _inputs:
        # compute INPUT gate from input and last/initial hidden state
        input_gate = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        # compute FORGET gate from input and last/initial hidden state
        forget_gate = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        # compute OUTPUT gate from input and last/initial hidden state
        output_gate = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        # compute memory cell candidate from input and last/initial hidden state
        memory_cell_candidate = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        # compute memory cell from last memory cell and memory cell candidate
        C = forget_gate * C + input_gate * memory_cell_candidate
        # compute hidden state from output gate and memory cell
        H = output_gate * nd.tanh(C)
        # compute output from hidden state
        Y = nd.dot(H, W_hy) + b_y
        _outputs.append(Y)

    return _outputs, H, C


if __name__ == '__main__':
    initial_state_h = nd.zeros(shape=(config.batch_size, config.hidden_dim))
    initial_state_c = nd.zeros(shape=(config.batch_size, config.hidden_dim))
    dump_data = [nd.random_normal(shape=(config.batch_size, config.input_dim)) for _ in range(config.num_steps)]

    parameters = get_parameters()
    _outputs, final_state, memory_cell = lstm(dump_data, initial_state_h, initial_state_c, *parameters)

    print(_outputs, final_state, memory_cell)
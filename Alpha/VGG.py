from mxnet.gluon import nn
from mxnet import gluon


class Config:
    def __init__(self):
        self.input_dim = 128
        self.hidden_dim = 256
        self.output_dim = 10
        self.full_connection_dim = 4096
        self.dropout_probability = 0.5

        self.std = 1e-2

        self.learning_rate = 1e-2
        self.num_epochs = 5
        self.batch_size = 32


config = Config()


def vgg_block(num_convs, channels):
    out = nn.Sequential()

    for _ in range(num_convs):
        out.add(nn.Conv2D(channels=channels, kernel_size=3,
                          padding=1, activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))

    return out


class vgg:
    def __init__(self, architecture):
        self.net = nn.Sequential()
        with self.net.name_scope():
            vgg_block_stack = nn.Sequential()
            for (num_convs, channels) in architecture:
                vgg_block_stack.add(vgg_block(num_convs=num_convs,
                                              channels=channels))
            self.net.add(vgg_block_stack)

            self.net.add(
                nn.Flatten(),
                nn.Dense(config.full_connection_dim, activation='relu'),
                nn.Dropout(config.dropout_probability),
                nn.Dense(config.full_connection_dim, activation='relu'),
                nn.Dropout(config.dropout_probability),
                nn.Dense(config.output_dim)
            )


# vgg_model = vgg()
# vgg_model.net.initialize()
#
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(vgg_model.net.collect_params(),
#                         'sgd', {'learning_rate': 0.05})

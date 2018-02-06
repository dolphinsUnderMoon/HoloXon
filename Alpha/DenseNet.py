from mxnet.gluon import nn
from mxnet import nd


class Config:
    def __init__(self):
        self.num_classes = 10

        self.std = 1e-2

        self.learning_rate = 1e-2
        self.num_epochs = 5
        self.batch_size = 32
        self.verbose = False


config = Config()


def conv_block(channels):
    block = nn.Sequential()
    block.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(channels, kernel_size=3, padding=1)
    )

    return block


class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.net = nn.Sequential()
            for i in range(layers):
                self.net.add(
                    conv_block(growth_rate)
                )

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)

        return x


class TransitionBlock(nn.Block):
    def __init__(self, channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)

        with self.name_scope():
            self.net = nn.Sequential()
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(channels, kernel_size=1),
                nn.AvgPool2D(pool_size=2, strides=2)
            )

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Block):
    def __init__(self, initial_channels, growth_rate, block_layers, num_classes, **kwargs):
        super(DenseNet, self).__init__(**kwargs)

        with self.name_scope():
            self.net = nn.Sequential()

            # first block
            self.net.add(
                nn.Conv2D(initial_channels, kernel_size=7,
                          strides=2, padding=3),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            )

            # dense blocks
            channels = initial_channels
            for i, layers in enumerate(block_layers):
                self.net.add(
                    DenseBlock(layers, growth_rate=growth_rate)
                )

                channels += layers * growth_rate
                if i != len(block_layers) - 1:
                    self.net.add(
                        TransitionBlock(channels=channels//2)
                    )

            # last block
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.AvgPool2D(pool_size=1),
                nn.Flatten(),
                nn.Dense(num_classes)
            )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    # dblk = DenseBlock(2, 10)
    # dblk.initialize()
    #
    # x = nd.random.uniform(shape=(4, 3, 8, 8))
    # print(dblk(x).shape)

    # dblk = TransitionBlock(10)
    # dblk.initialize()
    #
    # x = nd.random.uniform(shape=(4, 3, 8, 8))
    # print(dblk(x).shape)

    net = DenseNet(64, 32, [6, 12, 24, 16], 10)
    net.initialize()

    x = nd.random.uniform(shape=(4, 3, 96, 96))
    y = net(x)
    print(y.shape)
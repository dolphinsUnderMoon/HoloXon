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


class ResidualBlock(nn.Block):
    def __init__(self, channels, is_same_shape=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)

        self.is_same_shape = is_same_shape
        strides = 1 if self.is_same_shape else 2

        self.conv_1 = nn.Conv2D(channels=channels, kernel_size=3,
                                strides=strides, padding=1)
        self.bn_1 = nn.BatchNorm()
        self.conv_2 = nn.Conv2D(channels=channels, kernel_size=3,
                                padding=1)
        self.bn_2 = nn.BatchNorm()

        if not self.is_same_shape:
            self.conv_3 = nn.Conv2D(channels=channels, kernel_size=1,
                                    strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn_1(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))

        if not self.is_same_shape:
            x = self.conv_3(x)

        return nd.relu(out + x)


class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        self.verbose = verbose

        with self.name_scope():
            # block 1
            block_1 = nn.Conv2D(64, kernel_size=7, strides=2)

            # block 2
            block_2 = nn.Sequential()
            block_2.add(
                nn.MaxPool2D(pool_size=3, strides=2),
                ResidualBlock(64),
                ResidualBlock(64)
            )

            # block 3
            block_3 = nn.Sequential()
            block_3.add(
                ResidualBlock(128, is_same_shape=False),
                ResidualBlock(128)
            )

            # block 4
            block_4 = nn.Sequential()
            block_4.add(
                ResidualBlock(256, is_same_shape=False),
                ResidualBlock(256)
            )

            # block 5
            block_5 = nn.Sequential()
            block_5.add(
                ResidualBlock(512, is_same_shape=False),
                ResidualBlock(512)
            )

            # block 6
            block_6 = nn.Sequential()
            block_6.add(
                nn.AvgPool2D(pool_size=3),
                nn.Dense(num_classes)
            )

            self.net = nn.Sequential()
            self.net.add(block_1, block_2, block_3, block_4, block_5, block_6)

    def forward(self, x):
        out = x
        for i, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print('Block %d output\'s shape: %s' % (i + 1, out.shape))

        return out


if __name__ == '__main__':
    # blk = ResidualBlock(3)
    # blk.initialize()
    #
    # x = nd.random.uniform(shape=(4, 3, 6, 6))
    # print(blk(x).shape)

    net = ResNet(10, verbose=True)
    net.initialize()

    x = nd.random.uniform(shape=(4, 3, 96, 96))
    y = net(x)

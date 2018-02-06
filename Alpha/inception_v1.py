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


class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)

        self.version = "Inception V1"

        # path 1
        self.path_1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')

        # path 2
        self.path_2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
        self.path_2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,
                                       activation='relu')

        # path 3
        self.path_3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
        self.path_3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2,
                                       activation='relu')

        # path 4
        self.path_4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
        self.path_4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        path_1 = self.path_1_conv_1(x)
        path_2 = self.path_2_conv_3(self.path_2_conv_1(x))
        path_3 = self.path_3_conv_5(self.path_3_conv_1(x))
        path_4 = self.path_4_conv_1(self.path_4_pool_3(x))

        return nd.concat(path_1, path_2, path_3, path_4, dim=1)


class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)

        self.verbose = verbose

        with self.name_scope():
            # block 1
            block_1 = nn.Sequential()
            block_1.add(
                nn.Conv2D(64, kernel_size=7, strides=2,
                          padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 2
            block_2 = nn.Sequential()
            block_2.add(
                nn.Conv2D(64, kernel_size=1),
                nn.Conv2D(192, kernel_size=3, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 3
            block_3 = nn.Sequential()
            block_3.add(
                Inception(64, 96, 128, 16,32, 32),
                Inception(128, 128, 192, 32, 96, 64),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 4
            block_4 = nn.Sequential()
            block_4.add(
                Inception(192, 96, 208, 16, 48, 64),
                Inception(160, 112, 224, 24, 64, 64),
                Inception(128, 128, 256, 24, 64, 64),
                Inception(112, 144, 288, 32, 64, 64),
                Inception(256, 160, 320, 32, 128, 128),
                nn.MaxPool2D(pool_size=3, strides=2)
            )

            # block 5
            block_5 = nn.Sequential()
            block_5.add(
                Inception(256, 160, 320, 32, 128, 128),
                Inception(384, 192, 384, 48, 128, 128),
                nn.AvgPool2D(pool_size=2)
            )

            # block 6
            block_6 = nn.Sequential()
            block_6.add(
                nn.Flatten(),
                nn.Dense(num_classes)
            )

            self.net = nn.Sequential()
            self.net.add(block_1, block_2, block_3, block_4, block_5, block_6)

    def forward(self, x):
        out = x
        for i, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print("Block %d output\'s shape: %s" % (i + 1, out.shape))

        return out


if __name__ == '__main__':
    # inception_v1_test = Inception(64, 96, 128, 16, 32, 32)
    # inception_v1_test.initialize()
    #
    # x = nd.random.uniform(shape=(32, 3, 64, 64))
    # print(inception_v1_test(x).shape)

    net = GoogLeNet(10, verbose=True)
    net.initialize()

    x = nd.random.uniform(shape=(4, 3, 96, 96))
    y = net(x)

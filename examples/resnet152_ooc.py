import chainer
import math

import chainer.functions as F
import chainer.links as L
import cupy.cuda.stream as stream

class BottleNeckA_OOC(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA_OOC, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True, forget_x=True),
            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x, test=not train), forget_x=True))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train), forget_x=True))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train), forget_x=True))
        h2 = self.conv4(F.relu(self.bn4(x, test=not train), forget_x=True))

        y = h1 + h2
        h1.forget()
        h2.forget()

        return y


class BottleNeckB_OOC(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB_OOC, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.relu(self.bn1(x, test=not train), forget_x=True))
        h = self.conv2(F.relu(self.bn2(h, test=not train), forget_x=True))
        h = self.conv3(F.relu(self.bn3(h, test=not train), forget_x=True))

        y = h + x
        h.forget()

        return y


class Block_OOC(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block_OOC, self).__init__()
        links = [('a', BottleNeckA_OOC(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB_OOC(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train, stream=None):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)
            x.set_end_of_sub_graph(stream=stream)

        return x


class ResNet152_OOC(chainer.Chain):
    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet152_OOC, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64, forget_x=True),
            res2=Block_OOC(3, 64, 64, 256, 1),
            res3=Block_OOC(8, 256, 128, 512),
            res4=Block_OOC(36, 512, 256, 1024),
            res5=Block_OOC(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True
        self.disable_swapout_params()
        self.stream = None

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h, forget_x=True), 3, stride=2, forget_x=True)
        h.set_end_of_sub_graph(stream=self.stream)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.stream is None:
            self.stream = stream.Stream(non_blocking=True)

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            return loss
        else:
            return h

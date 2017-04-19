import chainer
import chainer.functions as F
import chainer.links as L


# copy-pasted from http://taichitary.hatenablog.com/entry/2017/02/17/232935
class AlexNet(chainer.Chain):

    input_size = 227

    def __init__(self):
        super(AlexNet, self).__init__(
            conv1 = L.Convolution2D(None, 96, 11, stride=4),
            conv2 = L.Convolution2D(None, 256, 5, pad=2),
            conv3 = L.Convolution2D(None, 384, 3, pad=1),
            conv4 = L.Convolution2D(None, 384, 3, pad=1),
            conv5 = L.Convolution2D(None, 256, 3, pad=1),
            fc6 = L.Linear(None, 4096),
            fc7 = L.Linear(None, 4096),
            fc8 = L.Linear(None, 10))

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        #h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.conv5(h))
        print(h.data.shape)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = F.relu(self.fc8(h))

        return h

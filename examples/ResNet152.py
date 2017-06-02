import chainer
import math
import numpy

import chainer.functions as F
import chainer.links as L

from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList

use_cuda = True

if (use_cuda):
    import cupy as xp
else:
    import numpy as xp

import cupy.cuda.nvtx as nvtx
import cupy.cuda.runtime as runtime

import time
from time import sleep

############################################################

class BottleNeckA_ref(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA_ref, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),

            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x, test=not train)))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train)))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train)))

        h2 = self.conv4(F.relu(self.bn4(x, test=not train)))

        return h1 + h2


class BottleNeckB_ref(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB_ref, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.relu(self.bn1(x, test=not train)))
        h = self.conv2(F.relu(self.bn2(h, test=not train)))
        h = self.conv3(F.relu(self.bn3(h, test=not train)))

        return h + x


class Block_ref(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block_ref, self).__init__()
        links = [('a', BottleNeckA_ref(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB_ref(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class ResNet152_ref(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet152_ref, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block_ref(3, 64, 64, 256, 1),
            res3=Block_ref(8, 256, 128, 512),
            res4=Block_ref(36, 512, 256, 1024),
            res5=Block_ref(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return h

############################################################


model = ResNet152_ref()
opt = optimizers.SGD()
opt.setup(model)
nbatch=10
num_loop = 10

# random data
x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable( xp.asarray(x) )

x0 = Variable( xp.zeros_like( x.data ) )
x0.copydata(x)
x0.name = "x0"
x0.cleargrad()

label = xp.zeros((nbatch), dtype=xp.int32)
for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable( xp.asarray(label) )

if (use_cuda):
    x0.to_gpu()
    label.to_gpu()
    model.to_gpu()

accum_t_f = 0
accum_t_b = 0

for loop in range(0, num_loop):
    
    runtime.deviceSynchronize()
    nvtx.RangePush("Run: {}".format(loop), loop)
    t0 = time.clock()
    
    model.cleargrads()

    nvtx.RangePush("Forward",1)
    loss = model(x0, label)
    runtime.deviceSynchronize()
    nvtx.RangePop()
    t1 = time.clock()
    
    print 'loop:{}, loss:{}'.format(loop, loss.data)
    
    nvtx.RangePush("Backward & Update",2)
    loss.backward()
    opt.update()
    runtime.deviceSynchronize()
    nvtx.RangePop()
    
    runtime.deviceSynchronize()
    nvtx.RangePop()
    t2 = time.clock()

    t_f = t1 - t0
    t_b = t2 - t1

    if loop >= 2:
        accum_t_f += t_f
        accum_t_b += t_b

ave_t_f = accum_t_f / (num_loop - 2)
ave_t_b = accum_t_b / (num_loop - 2)

print("nbatch:{}".format(nbatch))
print("average time: total:{}, forward:{}, backward:{}".format(ave_t_f+ave_t_b, ave_t_f, ave_t_b))

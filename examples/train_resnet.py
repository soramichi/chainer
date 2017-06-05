import chainer
import numpy
import time

import numpy as np
import cupy.cuda.runtime as runtime

from chainer import cuda, optimizers, serializers
from chainer import Variable
from resnet152 import ResNet152

use_cupy = True
xp = cuda.cupy if use_cupy else np

model = ResNet152()
opt = optimizers.SGD()
opt.setup(model)
nbatch = 20
num_loop = 10

# set up random data
x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable(xp.asarray(x))
label = xp.zeros((nbatch), dtype=xp.int32)

for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable(xp.asarray(label))

if use_cupy:
    x.to_gpu()
    label.to_gpu()
    model.to_gpu()

# measurement
accum_t_f = 0
accum_t_b = 0

for loop in range(0, num_loop):
    model.cleargrads()

    # t0: before each iteration
    t0 = time.clock()

    loss = model(x, label)

    # t1: between each forward and backward
    t1 = time.clock()
    
    print('loop:{}, loss:{}'.format(loop, loss.data))

    loss.backward()
    opt.update()

    # t2: after each iteration
    t2 = time.clock()

    t_f = t1 - t0
    t_b = t2 - t1

    if loop >= 2:
        accum_t_f += t_f
        accum_t_b += t_b


# print the results
ave_t_f = accum_t_f / (num_loop - 2)
ave_t_b = accum_t_b / (num_loop - 2)

print("average total: {}".format(ave_t_f + ave_t_b))
print("average forward: {}".format(ave_t_f))
print("average backward: {}".format(ave_t_b))

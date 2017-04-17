from __future__ import print_function
import argparse

import math
import numpy
import random
import functools

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG

model = None
error_rate = 0

def get_index(s, index):
    if len(s) == 1:
        return (int(index % s[0]), )
    else:
        tmp = functools.reduce(lambda a,b: a*b, s[1:], 1)
        return (int(index / tmp), ) + get_index(s[1:], index % tmp)

def inject_random_error(trainer):
    global model, error_rate

    total_size = 0
    total_len = 0
    for l in model.predictor._children:
        for n, p in model.predictor.__dict__[l].namedparams():
            if p.name == "W":
                total_size += p.data.size * 4 # assume float32
                total_len += p.data.size

    #print(total_size, total_len)
    errored_bits = int(math.ceil(total_size * 8 * error_rate))
    target = numpy.sort(numpy.random.permutation(total_len)[0:errored_bits])
    #print(target)

    """
    # to ensure that the paremeters are really changed in the model
    for l in model.predictor._children:
        for n, p in model.predictor.__dict__[l].namedparams():                
            if p.name == "W":
                model.predictor.__dict__[l].setparam(n[1:], chainer.cuda.cupy.zeros(p.data.shape, order='C', dtype=chainer.cuda.cupy.float32))
    """

    len_so_far = 0
    len_old = 0
    t = 0
    for l in model.predictor._children:
        for n, p in model.predictor.__dict__[l].namedparams():                
            if p.name == "W":
                modified = False
                len_old = len_so_far
                len_so_far += p.data.size
                W = chainer.cuda.to_cpu(p.data)

                # target must be sorted
                while t < len(target) and target[t] < len_so_far:
                    index = target[t] - len_old
                    W[get_index(W.shape, index)] = random.random()
                    modified = True
                    t += 1
  
                # because p is a copy, it needs to be written back
                if modified:
                    model.predictor.__dict__[l].setparam(n[1:], chainer.cuda.to_gpu(W))

    """
    for t in target:
        len_so_far = 0
        len_old = 0
        for l in model.predictor._children:
            for n, p in model.predictor.__dict__[l].namedparams():                
                if p.name == "W":
                    len_so_far += len(p.data)
                    if len_so_far > t:
                        index = t - len_old
                        p.data[get_index(p.data.shape, index)] = random.random()

                        # because p is a copy, it needs to be written back
                        model.predictor.__dict__[l].setparam(n[1:], p.data)
                        break # for p
                    else:
                        len_old = len_so_far
    """

def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--error', '-E', type=float, default=0,
                        help='Error rate')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Error rate: {}'.format(args.error))
    print('')

    global error_rate
    error_rate = args.error

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    global model
    model = L.Classifier(models.VGG.VGG(class_labels))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    trainer.extend(inject_random_error, trigger=(1, 'iteration'))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()

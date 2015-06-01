# this profiles the nets, to look at speed issues

%load_ext autoreload
%autoreload 2

import numpy as np
from tlstm.datahandler import DataHandler
from tlstm import sgd as optimizer
import optparse
import cPickle as pickle
import conf_gradcheck as opts
import collections
import sys
from copy import deepcopy
import cProfile


test_mode = False
net_to_test = 'both'

if test_mode:
    opts.wvecDim = 5
    opts.middleDim = 8
    opts.sharedDim = 6
    opts.sentenceDim = opts.middleDim
    #opts.reg = 0
    #opts.rho = 0
    mult_factor = 1#50

# the relative error for gradients
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1, np.abs(x) + np.abs(y))))

# ensure the options are valid
assert opts.megabatch_size % opts.minibatch_size == 0
assert type(opts.data_type) == str
opts.data_type = opts.data_type.lower()
assert opts.data_type in ['vgg16','vgg19','both']

# set opts that have only one possible value
opts.numWords = 33540
opts.imageDim = 4096
if opts.data_type == 'both':
    opts.imageDim *= 2

if not 'b' in locals():
    # instantiate the data handler
    dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)
    # grab a batch
    dh.cur_iteration = 0
    b = dh.nextBatch()
    if test_mode:
        opts.imageDim = 5
        for x in b:
            x[0] = x[0][:opts.imageDim]
            x[0] = x[0] * mult_factor
from tlstm.tlstm import TLSTM
from tlstm.twin import Twin

if test_mode:
    opts.imageDim = 5

net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), opts.reg)

net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), 0, net2, root=opts.root)

from tlstm.tlstm_theano import TLSTM
net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), 0, net2, root=opts.root)

if test_mode:
    net1.L = net1.L[:,:opts.wvecDim] * mult_factor
    net1.stack[0] = net1.L

if net_to_test == '1':
    stack = net1.stack
    names = net1.names
elif net_to_test == '2':
    stack = net2.stack
    names = net2.names
else:
    stack = net1.stack + net2.stack
    names = net1.names + net2.names

# profiling
def iterate_costAndGrad(n):
    for i in range(n):
        print i
        net1.costAndGrad(b)

print 'Beginning profile'

#cProfile.run('iterate_costAndGrad(10)')

# the T-LSTM's forward/backward props take by far the longest, so let's
# try profiling them individually
#from line_profiler import LineProfiler
#profiler = LineProfiler()
#profiler.add_function(net1.forwardProp)
#profiler.enable_by_count()






try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner

@do_profile(follow=[net1.forwardProp])
def cab(b):
    net1.costAndGrad(b)

cab(b)

@do_profile(follow=[net1.backProp])
def cab(b):
    net1.costAndGrad(b)

cab(b)

%load_ext autoreload
%autoreload 2

import numpy as np
from tlstm.datahandler import DataHandler
from tlstm import sgd as optimizer
import optparse
import cPickle as pickle
import conf as opts

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

# instantiate the data handler
dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)

dh.cur_iteration = 0
from tlstm.tlstm import TLSTM
from tlstm.twin import Twin
# instantiate the second 'layer'
net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), opts.reg)

# instantiate the first 'layer'
net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), opts.rho, net2)

# instantiate the SGD
sgd = optimizer.SGD(net1, opts.alpha, dh, optimizer='sgd')

sgd.run()

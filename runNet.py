### NOTE: IF YOU ARE NOT USING IPYTHON, REMOVE THE NEXT TWO LINES
%load_ext autoreload
%autoreload 2

import numpy as np
from tlstm.datahandler import DataHandler
from datetime import datetime
import optparse
import cPickle as pickle
import conf as opts
from tlstm.tlstm import TLSTM
from tlstm.twin import Twin
from tlstm import sgd as optimizer

# ensure the options are valid
assert opts.megabatch_size % opts.minibatch_size == 0
assert type(opts.data_type) == str
opts.data_type = opts.data_type.lower()
assert opts.data_type in ['vgg16','vgg19','both']

test_mode = True

# set opts that have only one possible value
opts.numWords = 33540
opts.imageDim = 4096
if opts.data_type == 'both':
    opts.imageDim *= 2

# instantiate the data handler
dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)

dh.cur_iteration = 0

if opts.saved_model is not None:
	params = np.load(opts.saved_model)

# instantiate the second 'layer'
net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), 0, params=params)
#net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), 0)

# instantiate the first 'layer'
net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), 0, net2, root=opts.root, params=params)

#net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), 0, net2)

# instantiate the SGD
model_filename = "models/m_" + datetime.now().strftime("%m%d_%H%M%S") + "_%s"
log_filename = "logs/m_" + datetime.now().strftime("%m%d_%H%M%S.log")
sgd = optimizer.SGD(net1, model_filename, opts.alpha, dh, optimizer=opts.optimizer, logfile=log_filename)

#sgd = optimizer.SGD(net1, 1e-5, dh, optimizer='sgd')

sgd.run()
sgd.save_checkpoint("final")
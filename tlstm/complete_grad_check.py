# This performs a grad check on the entire network, exploiting the
# fact that both the first and second level networks store their stacks
# by reference.

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
from __future__ import print_function

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

if not 'dh' in locals():
    # instantiate the data handler
    dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)

    dh.cur_iteration = 0
    # grab a batch
if not 'b' in locals():
    b = dh.nextBatch()

from tlstm.tlstm import TLSTM
from tlstm.twin import Twin

net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), opts.reg)

net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), 0, net2, root=opts.root)

stack = net1.stack + net2.stack
grads = net1.grads + net2.grads
names = net1.names + net2.names

cost, _ = net1.costAndGrad(b)

epsilon = 1e-4
comp_grads = []
# check L first
L = net1.stack[0]
dL = net1.grads[0]
this_grad = collections.defaultdict(net1.defaultVec)
print('Checking dL')
for n,i in enumerate(dL.iterkeys()):
    for j in xrange(L.shape[1]):
        print('\tprog: %04i-%04i / %04i-%04i'%(n,j,len(dL),L.shape[1]), end="\r")
        sys.stdout.flush()
        L[i,j] += epsilon / 2
        costP, _ = net1.costAndGrad(b, test=True)
        L[i,j] -= epsilon / 2
        costN, _ = net1.costAndGrad(b, test=True)
        this_grad[i][j] = (costP - costN) / epsilon
        L[i,j] += epsilon / 2
comp_grads.append(this_grad)

for W, name in zip(stack[1:], names[1:]):
    print('Checking d%s'%name)
    W = W[..., None]
    this_grad = np.zeros_like(W)
    for i in xrange(W.shape[0]):
        for j in xrange(W.shape[1]):
            print('\tprog: %04i-%04i / %04i-%04i'%(i,j,W.shape[0],W.shape[1]), end="\r")
            sys.stdout.flush()
            W[i,j] += epsilon / 2
            costP, _ = net1.costAndGrad(b, test=True)
            W[i,j] -= epsilon / 2
            costN, _ = net1.costAndGrad(b, test=True)
            this_grad[i,j] = (costP - costN) / epsilon
            W[i,j] += epsilon / 2
    comp_grads.append(this_grad)

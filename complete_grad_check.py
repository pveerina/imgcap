from __future__ import print_function
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
from copy import deepcopy


test_mode = True
net_to_test = '2'

if test_mode:
    opts.wvecDim = 5
    opts.middleDim = 8
    opts.sharedDim = 6
    opts.sentenceDim = opts.middleDim
    #opts.reg = 0
    #opts.rho = 0
    mult_factor = 1#50
    opts.numLayers = 1

# the relative error for gradients
def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-6, np.abs(x) + np.abs(y))))

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

cost, _ = net1.costAndGrad(b, testCost=True)


# # 'update' the parameters, just for testing
# update = net2.grads
# net2.updateParams(-1e-5, update)

if net_to_test == '1':
    grads = deepcopy(net1.grads)
elif net_to_test == '2':
    grads = deepcopy(net2.grads)
else:
    grads = deepcopy(net1.grads) + deepcopy(net2.grads)

epsilon = 1e-5
comp_grads = []
if net_to_test != '2':
    # check L first
    L = net1.stack[0]
    dL = net1.grads[0]
    this_grad = collections.defaultdict(net1.defaultVec)
    print('Checking dL')
    cnt = 0
    for n,i in enumerate(dL.iterkeys()):
        for j in xrange(L.shape[1]):
            cnt+=1
            if cnt == len(dL)*L.shape[1]:
                print('\tprog: %6i / %6i'%(cnt,len(dL)*L.shape[1]))
            else:
                print('\tprog: %6i / %6i'%(cnt,len(dL)*L.shape[1]), end="\r")
            sys.stdout.flush()
            L[i,j] += epsilon / 2
            costP, _, _, _ = net1.costAndGrad(b, test=True)
            L[i,j] -= epsilon / 2
            costN, _, _, _ = net1.costAndGrad(b, test=True)
            this_grad[i][j] = (costP - costN) / epsilon
            L[i,j] += epsilon / 2
    comp_grads.append(this_grad)

    for W, name in zip(stack[1:], names[1:]):
        print('Checking d%s'%name)
        cnt = 0
        W = W[..., None]
        this_grad = np.zeros_like(W)
        for i in xrange(W.shape[0]):
            for j in xrange(W.shape[1]):
                cnt += 1
                if cnt == W.size:
                    print('\tprog: %6i / %6i'%(cnt,W.size))
                else:
                    print('\tprog: %6i / %6i'%(cnt,W.size), end="\r")
                sys.stdout.flush()
                W[i,j] += epsilon / 2
                costP, _, _, _ = net1.costAndGrad(b, test=True)
                W[i,j] -= epsilon
                costN, _, _, _ = net1.costAndGrad(b, test=True)
                this_grad[i,j] = (costP - costN) / epsilon
                W[i,j] += epsilon / 2
        comp_grads.append(this_grad)
else:
    for W, name in zip(stack, names):
        print('Checking d%s'%name)
        cnt = 0
        W = W[..., None]
        this_grad = np.zeros_like(W)
        for i in xrange(W.shape[0]):
            for j in xrange(W.shape[1]):
                cnt += 1
                if cnt == W.size:
                    print('\tprog: %6i / %6i'%(cnt,W.size))
                else:
                    print('\tprog: %6i / %6i'%(cnt,W.size), end="\r")
                sys.stdout.flush()
                W[i,j] += epsilon / 2
                costP, _, _, _ = net1.costAndGrad(b, test=True)
                W[i,j] -= epsilon
                costN, _, _, _ = net1.costAndGrad(b, test=True)
                this_grad[i,j] = (costP - costN) / epsilon
                W[i,j] += epsilon / 2
        comp_grads.append(this_grad)
if not test_mode:
    np.savez('grad_checks/grads.npz',[np.array(grads[0].values())] + grads[1:])
    np.savez('grad_checks/comp_grads.npz',[np.array(comp_grads[0].values())] + comp_grads[1:])

gradD = dict()
for i,j,k in zip(names, grads, comp_grads):
    gradD[i] = [j,k]

for k in sorted(gradD.keys()):
    try:
        a, cb = gradD[k]
        a = a.squeeze()
        cb = cb.squeeze()
        osz = [a.shape, cb.shape]
        error = rel_error(a,cb)
        print('%s : %g [%s vs %s]'%(k, error * (error > 1e-5), str(osz[0]), str(osz[1])))
    except:
        a = np.array(gradD[k][0].values())
        cb = np.array(gradD[k][1].values())
        a = a.squeeze()
        cb = cb.squeeze()
        osz = [a.shape, cb.shape]
        error = rel_error(a,cb)
        print('%s : %g [%s vs %s]'%(k, error * (error > 1e-5), str(osz[0]), str(osz[1])))

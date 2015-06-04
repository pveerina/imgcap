from __future__ import print_function
#%load_ext autoreload
#%autoreload 2

import numpy as np
import sys

def test(net, dh):
    # accepts a neural net, a data handler, and the options. Returns
    # two measure of accuracy
    b = dh.nextBatch(test=True)
    baseline = 1./len(b)
    costs = []
    xs = []
    ys = []
    while b != -1:
        cost, total, xst, yst = net.costAndGrad(b, test=True)
        xs += list(xst)
        ys += list(yst)
        if len(dh.test_minibatch_queue):
            print('%5i rem cost:%6.3f'%(len(dh.test_minibatch_queue), cost), end="\r")
            sys.stdout.flush()
        else:
            print('%5i rem cost:%6.3f'%(len(dh.test_minibatch_queue), cost))
            sys.stdout.flush()
        costs.append(cost)
        b = dh.nextBatch(test=True)
    dp = np.argmax(np.dot(np.array(xs),np.array(ys).T),1)
    score2 = np.sum(dp==np.array(range(len(dp)))) * 1./len(dp)
    mc = np.mean(costs)
    ms = score2
    print('Mean cost: %g, R@1: %g'%(mc, ms))
    return mc, ms

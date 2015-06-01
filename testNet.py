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
    scores = []
    while b != -1:
        cost, total, xs, ys = net.costAndGrad(b, test=True)
        dp = np.dot(xs, ys.T)
        res = np.argmax(dp)
        score2 = np.sum(np.argmax(dp,1)==np.array(range(len(dp)))) * 1./len(dp)
        if len(dh.test_minibatch_queue):
            print('%i rem cost:%g score:%g'%(len(dh.test_minibatch_queue), cost, score2), end="\r")
            sys.stdout.flush()
        else:
            print('%i rem cost:%6.3f score:%g'%(len(dh.test_minibatch_queue), cost, score2))
            sys.stdout.flush()
        costs.append(cost)
        scores.append(score2)
        b = dh.nextBatch(test=True)
    mc = np.mean(costs)
    ms = np.mean(scores)
    print('Mean cost: %g, Mean score: %g [baseline: %g]'%(mc, ms, baseline))
    return mc, ms

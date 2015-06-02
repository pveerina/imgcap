import numpy as np
import random
from testNet import test
import time

def printTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)

class SGD:

    def __init__(self, model, modelfilename, alpha=1e-2,dh=None, optimizer='sgd', logfile=None, test_inc=1000, save_on_interrupt=False):
        # dh = instance of data handler
        self.model1 = model
        self.model2 = model.topLayer
        totparams = np.sum([x.size for x in self.model1.stack]) + np.sum([x.size for x in self.model2.stack])
        print '%i total parameters'%(totparams)
        self.dh = dh
        self.logfile = logfile
        self.model_filename = modelfilename
        print "initializing SGD"
        assert self.model1 is not None, "Must define a function to optimize"
        self.it = 0
        self.alpha = alpha # learning rate
        self.optimizer = optimizer
        self.test_inc = test_inc
        self.save_on_interrupt = save_on_interrupt
        if self.optimizer == 'sgd':
            print "Using sgd.."
        elif self.optimizer == 'adagrad':
            print "Using adagrad..."
            epsilon = 1e-8
            self.gradt1 = [epsilon + np.zeros(W.shape) for W in self.model1.stack]
            self.gradt2 = [epsilon + np.zeros(W.shape) for W in self.model2.stack]
        else:
            raise ValueError("Invalid optimizer")

        self.costt = []
        self.expcost = []

    def run(self):
        try:
            """
            Runs stochastic gradient descent with model as objective.
            """
            print "running SGD"
            start = time.time()
            self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))
            mbdata = self.dh.nextBatch()
            prev_megabatch = 0
            all_iter = 0
            self.dev_costs = []
            self.dev_scores = []
            while mbdata != None:
                if not self.test_inc == None:
                    if not all_iter % self.test_inc:
                        devco, devsco = test(self.model1, self.dh)
                        self.dev_costs.append(devco)
                        self.dev_scores.append(devsco)
                all_iter += 1
                self.it = self.dh.cur_iteration
                cost, _ = self.model1.costAndGrad(mbdata)
                grad1 = self.model1.grads
                grad2 = self.model2.grads
                if self.it > 1:
                    if cost > 6*self.expcost[-1]:
                        print 'Unusual cost observed, creating checkpoint...'
                        self.save_checkpoint('_UNUSUALCOST_iter_%i'%all_iter)
                        self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))
                if np.isfinite(cost):
                    if self.it > 1:
                        self.expcost.append(.01*cost + .99*self.expcost[-1])
                    else:
                        self.expcost.append(cost)
                if self.optimizer == 'sgd':
                    update1 = grad1
                    update2 = grad2
                    scale = -self.alpha

                elif self.optimizer == 'adagrad':

                    self.gradt1[1:] = [gt+g**2
                            for gt,g in zip(self.gradt1[1:],grad1[1:])]
                    update =  [g*(1./np.sqrt(gt))
                            for gt,g in zip(self.gradt1[1:],grad1[1:])]
                    # handle dictionary separately
                    dL = grad1[0]
                    dLt = self.gradt1[0]
                    for j in dL.iterkeys():
                        dLt[j] = dLt[j,:] + dL[j]**2
                        dL[j] = dL[j] * (1./np.sqrt(dLt[j,:]))
                    update1 = [dL] + update
                    #
                    # Now perform it for network 2
                    #
                    self.gradt2 = [gt+g**2
                            for gt,g in zip(self.gradt2,grad2)]
                    update2 =  [g*(1./np.sqrt(gt))
                            for gt,g in zip(self.gradt2,grad2)]
                    # handle dictionary separately

                    scale = -self.alpha

                # update params
                self.model1.updateParams(scale,update1,log=False)
                self.model2.updateParams(scale,update2)

                self.costt.append(cost)
                # compute time remaining
                cur = time.time()
                tdiff = (cur-start)
                timePerIter = tdiff * 1./all_iter
                timeRem = timePerIter * (self.dh.batchPerEpoch - self.it)
                if self.it%1 == 0:
                    msg = "Iter:%6d [%6i] (rem:%6d, %s so far, %s to next epoch) mbatch:%d epoch:%d cost=%7.4f, exp=%7.4f."%(self.it,all_iter,len(self.dh.minibatch_queue), printTime(tdiff), printTime(timeRem), self.dh.cur_megabatch, self.dh.cur_epoch, cost,self.expcost[-1])
                    print msg
                    if self.logfile is not None:
                        with open(self.logfile, "a") as logfile:
                            logfile.write(msg + "\n")
                mbdata = self.dh.nextBatch()
                if self.dh.cur_megabatch != prev_megabatch:
                    # checkpoint
                    prev_megabatch = self.dh.cur_megabatch
                    self.save_checkpoint("%d_epoch%i"%(prev_megabatch, self.dh.cur_epoch))
        except KeyboardInterrupt as ke:
            if self.save_on_interrupt:
                self.save_checkpoint('_INTERRUPT')
                self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))
        except FloatingPointError as fe:
            self.save_checkpoint('_FLOATING_POINT_ERROR_iter%i'%all_iter)
            self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))


    def save_checkpoint(self, checkpoint_name):
        param_dict = dict(zip(self.model1.names, self.model1.stack) + zip(self.model2.names, self.model2.stack))
        np.savez(self.model_filename%checkpoint_name, **param_dict)
        with open(self.model_filename%'dev_scores','w') as f:
            f.write(str(self.dev_scores))
        with open(self.model_filename%'dev_costs','w') as f:
            f.write(str(self.dev_costs))
        with open(self.model_filename%'train_costs','w') as f:
            f.write(str(self.expcost))
        self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))

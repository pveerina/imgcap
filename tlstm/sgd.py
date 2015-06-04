import numpy as np
import random
from testNet import test
import time

def printTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)

class SGD:

    def __init__(self, model, modelfilename, alpha=1e-2, dh=None, optimizer='sgd', logfile=None, test_inc=1000, save_on_interrupt=True):
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
        self.lr_decay = 0.95
        self.mu = .5
        self.mu_coeff = .03
        self.lr_step = 2500
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
            use_momentum = True
            vel1 = None
            vel2 = None
            while mbdata != None:
                try:
                    if not self.test_inc == None:
                        if not all_iter % self.test_inc:
                            devco, devsco = test(self.model1, self.dh)
                            self.dev_costs.append(devco)
                            self.dev_scores.append(devsco)
                    all_iter += 1
                    if all_iter == 1:
                        print 'Learning rate is %g'%(self.alpha)
                    if all_iter > 5 and not all_iter%self.lr_step:
                        self.alpha *= self.lr_decay
                        self.mu += ((1-self.mu)-.05)*self.mu_coeff
                        print 'Updating learning rate to %g'%(self.alpha)
                        print 'Updating momentum to %g'%(self.mu)
                    self.it = self.dh.cur_iteration
                    cost, _ = self.model1.costAndGrad(mbdata)
                    grad1 = self.model1.grads
                    grad2 = self.model2.grads
                    if all_iter > 1:
                        if cost > 10*self.expcost[-1]:
                            print 'Unusual cost observed, creating checkpoint...'
                            self.save_checkpoint('_UNUSUALCOST_iter_%i'%all_iter)
                            self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))
                    if np.isfinite(cost):
                        if self.it > 1:
                            self.expcost.append(.01*cost + .99*self.expcost[-1])
                        else:
                            self.expcost.append(cost)
                    if self.optimizer == 'sgd':
                        if use_momentum:
                            scale = -self.alpha/(1+self.mu)
                            if vel1 == None:
                                vel1 = [0] + [x * scale for x in grad1[1:]]
                                vel2 = [x * scale for x in grad2]
                            vel1[0] = grad1[0]
                            for j in vel1[0].iterkeys():
                                vel1[0][j] *= scale
                            for n,x in enumerate(grad1):
                                if n == 0:
                                    continue
                                vel1[n] *= self.mu
                                vel1[n] += scale * x
                            for n,x in enumerate(grad2):
                                vel2[n] *= self.mu
                                vel2[n] += scale * x
                            update1 = vel1
                            update2 = vel2
                            scale = 1
                        else:
                            scale = -self.alpha
                            update1 = grad1
                            update2 = grad2

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
                except FloatingPointError as fe:
                    print 'FLOATING POINT ERROR!'
                    print fe.message
                    print 'Saving checkpoint...'
                    self.save_checkpoint('_FLOATING_POINT_ERROR_iter%i'%all_iter)
                    self.dh.saveSets('/'.join(self.model_filename.split('/')[:-1]))
                    msg = "Iter:%6d [%6i] (rem:%6d, %s so far, %s to next epoch) mbatch:%d epoch:%d cost=%7.4f, exp=%7.4f. FLOATING POINT ERROR HERE"%(self.it,all_iter,len(self.dh.minibatch_queue), printTime(tdiff), printTime(timeRem), self.dh.cur_megabatch, self.dh.cur_epoch, cost,self.expcost[-1])
                    if self.logfile is not None:
                        with open(self.logfile, "a") as logfile:
                            logfile.write(msg + "\n")
                    print 'Discarding minibatch and parameter updates'
                    if use_momentum:
                        vel1 = None
                        vel2 = None
                    continue
                # update params
                #self.model1.updateParams(scale,update1,log=True)
                #self.model2.updateParams(scale,update2,log=True)
                self.model1.updateParams(scale,update1)
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
            print 'KEYBOARD INTERRUPT'
            if self.save_on_interrupt:
                self.save_checkpoint('_INTERRUPT')
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

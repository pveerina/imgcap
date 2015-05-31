import numpy as np
import random

class SGD:

    def __init__(self, model, modelfilename, alpha=1e-2,dh=None, optimizer='sgd', logfile=None):
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
        """
        Runs stochastic gradient descent with model as objective.
        """
        print "running SGD"
        mbdata = self.dh.nextBatch()
        prev_megabatch = 0
        while mbdata != None:
            self.it = self.dh.cur_iteration
            cost, _ = self.model1.costAndGrad(mbdata)
            grad1 = self.model1.grads
            grad2 = self.model2.grads
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
                ## ADAGRAD CURRENTLY DOESN'T WORK, SINCE THE GRADIENTS IN
                ## BOTH NETWORKS ARE REPRESENTED AS LISTS RATHER THAN
                ## MATRICES
                #
                # Perform update for network 1 first
                #
                # for gt1, g1 in zip(self.gradt[1:], grad1[1:]):
                #     if type(g1) == list:
                #         for gt2, g2 in zip(gt1, g1):
                #             if type(g2) == list:
                #                 for gt3, g3 in zip(gt2, g2):

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
            if self.it%1 == 0:
                msg = "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])
                print msg
                if self.logfile is not None:
                    with open(self.logfile, "a") as logfile:
                        logfile.write(msg + "\n")
            mbdata = self.dh.nextBatch()
            if self.dh.cur_megabatch != prev_megabatch:
                # checkpoint
                prev_megabatch = self.dh.cur_megabatch
                self.save_checkpoint("%d"%prev_megabatch)


    def save_checkpoint(self, checkpoint_name):
        param_dict = dict(zip(self.model1.names, self.model1.stack) + zip(self.model2.names, self.model2.stack))
        np.savez(self.model_filename%checkpoint_name, **param_dict)


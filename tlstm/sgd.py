import numpy as np
import random

class SGD:

    def __init__(self,model,alpha=1e-2,dh=None,
                 optimizer='sgd'):
        # dh = instance of data handler
        self.model1 = model
        self.model2 = model.topLayer
        self.dh = dh
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
                print "Iter %d : Cost=%.4f, ExpCost=%.4f."%(self.it,cost,self.expcost[-1])
            mbdata = self.dh.nextBatch()


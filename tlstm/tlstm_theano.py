'''
A theano implementation of the T-LSTM
'''

import theano.tensor as T
from theano import function
import numpy as np
import collections
import pdb
import os
#np.seterr(under='warn')
h, b = T.fvectors('h', 'b')
W, X = T.fmatrices('W', 'X')

dotvec = function([h,b], T.dot(h,b))

dot = function([W, h], T.dot(W, h))
#dotF = function([W, h], T.dot(W, h))
#dot = lambda W, h: dotF(W, h.squeeze())
dotW = function([W, X], T.dot(W,X))

layer = function([W, h, b], T.dot(W, h) + b)
#layerF = function([W, h, b], T.dot(W, h) + b)
#layer = lambda W, h, b: layerF(W, h.squeeze(), b.squeeze())
sigmoid = function([h], T.nnet.ultra_fast_sigmoid(h))
#sigmoidF = function([h], T.nnet.ultra_fast_sigmoid(h))
#sigmoid = lambda h: sigmoidF(h.squeeze())
tanh = function([h], T.tanh(h))
#tanhF = function([h], T.tanh(h))
#tanh = lambda h: tanhF(h.squeeze())
add = function([h, b], h+b)
#addF = function([h, b], h+b)
#add = lambda h, b: addF(h.squeeze(), b.squeeze())

class TLSTM:

    def __init__(self,wvecDim, middleDim, paramDim, numWords,mbSize=30, scale=1, rho=1e-4, topLayer = None, root=None, params=None):
        self.wvecDim = wvecDim
        self.middleDim = middleDim
        self.paramDim = paramDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.scale = scale
        self.defaultVec = lambda : np.zeros((wvecDim,), dtype='float32')
        self.rho = rho
        self.topLayer = topLayer
        self.root = root
        self.initParams(params)
    def initParams(self, params):
        # MAKE SURE THEY READ IN

        # Word vectors
        #self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)
        self.L = np.load(os.path.join(self.root, 'data','trees','Lmat.npy')).astype('float32')

        def ybt(sz=self.middleDim): # yield bias term
            return np.zeros((sz)).astype('float32')

        def yw(sz=(self.middleDim, self.wvecDim)): # yield input weights
            return 0.01*np.random.randn(sz[0], sz[1]).astype('float32')

        def ydbt(sz=self.middleDim): # yield bias term
            return np.zeros((sz)).astype('float32')

        def ydw(sz=(self.middleDim, self.wvecDim)): # yield input weights
            return 0.01*np.zeros(sz).astype('float32')

        md = self.middleDim
        pd = self.paramDim
        # Bias Terms
        self.bf, self.bi, self.bo, self.bu = ybt(), ybt(), ybt(), ybt()

        # Input Weights
        self.Wu, self.Wo, self.Wi, self.Wf = yw(), yw(), yw(), yw()

        # Left Hidden Weights
        mds = (md, md)
        self.Ui = [yw(mds) for j in xrange(pd)]
        self.Ul = [[yw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.Ur = [[yw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.Uo = [yw(mds) for j in xrange(pd)]
        self.Uu = [yw(mds) for j in xrange(pd)]

        # Right Hidden Weights
        self.Vi = [yw(mds) for j in xrange(pd)]
        self.Vl = [[yw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.Vr = [[yw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.Vo = [yw(mds) for j in xrange(pd)]
        self.Vu = [yw(mds) for j in xrange(pd)]

        # Gradients
        self.dbf, self.dbi, self.dbo, self.dbu = ydbt(), ydbt(), ydbt(), ydbt()
        self.dWu, self.dWo, self.dWi, self.dWf = ydw(), ydw(), ydw(), ydw()

        self.dUi = [ydw(mds) for j in xrange(pd)]
        self.dUl = [[ydw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.dUr = [[ydw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.dUo = [ydw(mds) for j in xrange(pd)]
        self.dUu = [ydw(mds) for j in xrange(pd)]

        # Right Hidden Weights
        self.dVi = [ydw(mds) for j in xrange(pd)]
        self.dVl = [[ydw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.dVr = [[ydw(mds) for j in xrange(pd)] for k in xrange(pd)]
        self.dVo = [ydw(mds) for j in xrange(pd)]
        self.dVu = [ydw(mds) for j in xrange(pd)]

        # construct the stack and the gradient stack (grads)
        self.names = ['L']
        self.stack = [self.L]
        self.grads = [None] # dL isn't defined until costAndGrad?

        self.stack.append(self.Wo)
        self.names.append('Wo')
        self.grads.append(self.dWo)

        for j in range(self.paramDim):
            self.stack.append(self.Uo[j])
            self.names.append('Uo%i'%j)
            self.grads.append(self.dUo[j])
        for j in range(self.paramDim):
            self.stack.append(self.Vo[j])
            self.names.append('Vo%i'%j)
            self.grads.append(self.dVo[j])

        self.stack.append(self.bo)
        self.names.append('bo')
        self.grads.append(self.dbo)

        self.stack.append(self.Wi)
        self.names.append('Wi')
        self.grads.append(self.dWi)

        for j in range(self.paramDim):
            self.stack.append(self.Ui[j])
            self.names.append('Ui%i'%j)
            self.grads.append(self.dUi[j])
        for j in range(self.paramDim):
            self.stack.append(self.Vi[j])
            self.names.append('Vi%i'%j)
            self.grads.append(self.dVi[j])

        self.stack.append(self.bi)
        self.names.append('bi')
        self.grads.append(self.dbi)

        self.stack.append(self.Wu)
        self.names.append('Wu')
        self.grads.append(self.dWu)

        for j in range(self.paramDim):
            self.stack.append(self.Uu[j])
            self.names.append('Uu%i'%j)
            self.grads.append(self.dUu[j])
        for j in range(self.paramDim):
            self.stack.append(self.Vu[j])
            self.names.append('Vu%i'%j)
            self.grads.append(self.dVu[j])

        self.stack.append(self.bu)
        self.names.append('bu')
        self.grads.append(self.dbu)

        self.stack.append(self.Wf)
        self.names.append('Wf')
        self.grads.append(self.dWf)

        for j in range(self.paramDim):
            for k in range(self.paramDim):
                self.stack.append(self.Ul[j][k])
                self.names.append('Ul%i_%i'%(j,k))
                self.grads.append(self.dUl[j][k])
        for j in range(self.paramDim):
            for k in range(self.paramDim):
                self.stack.append(self.Vl[j][k])
                self.names.append('Vl%i_%i'%(j,k))
                self.grads.append(self.dVl[j][k])
        for j in range(self.paramDim):
            for k in range(self.paramDim):
                self.stack.append(self.Ur[j][k])
                self.names.append('Ur%i_%i'%(j,k))
                self.grads.append(self.dUr[j][k])
        for j in range(self.paramDim):
            for k in range(self.paramDim):
                self.stack.append(self.Vr[j][k])
                self.names.append('Vr%i_%i'%(j,k))
                self.grads.append(self.dVr[j][k])

        self.stack.append(self.bf)
        self.names.append('br')
        self.grads.append(self.dbf)
        self.reg_idx = []
        for n,i in enumerate(self.names):
            if not i[0]=='b' and not i[0]=='L':
                self.reg_idx.append(n)
        self.l = []
        for j in range(self.paramDim):
            self.l.append(np.zeros((self.middleDim, 1), dtype='float32'))
        self.r = []
        for j in range(self.paramDim):
            self.r.append(np.zeros((self.middleDim, 1), dtype='float32'))


    def costAndGrad(self,mbdata,test=False, testCost=False):
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W-terms, U-terms, V-terms, bias terms
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns
           cost, correctArray, guessArray, total

        Note that mbdata is a tuple, consisting of an image vector and
        a tree.
        """
        cost = 0.0
        total = 0.0

        assert len(self.stack) == 9+(6+4*self.paramDim)*self.paramDim

        # Zero gradients
        for cgrad in self.grads[1:]:
            cgrad[:] = 0

        self.dL = collections.defaultdict(self.defaultVec)
        self.grads[0] = self.dL
        # Forward prop each tree in minibatch
        newmbdata = []
        for imgvec, tree in mbdata:
            tot = self.forwardProp(tree.root)
            total += tot
            newmbdata.append((imgvec, tree.root.hActs2))

        if test:
            cost = self.topLayer.costAndGrad(newmbdata, test=True, testCost=testCost)
        else:
            if testCost:
                cost, error = self.topLayer.costAndGrad(newmbdata, testCost=testCost)
            else:
                cost, error = self.topLayer.costAndGrad(newmbdata)

        cost *= self.scale

        # Add L2 Regularization
        for i in self.reg_idx:
            cost += (self.rho/2)*np.sum(self.stack[i]**2)

        if test:
            return cost, total

         # Back prop each tree in minibatch
        for n, (_, tree) in enumerate(mbdata):
            cerror = error[n]
            if not len(cerror.shape) > 1:
                cerror = cerror[:, np.newaxis]
            self.backProp(tree.root, cerror)

        # scale cost and grad by mb size
        for v in self.dL.itervalues():
            v *=self.scale

        for i in self.grads[1:]:
            i *= self.scale
        for i in self.reg_idx:
            self.grads[i] += self.stack[i]*self.rho

        return cost, total

    def forwardProp(self,node):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
        x = np.reshape(self.L[node.word,:], (self.wvecDim, 1)).squeeze()
        if node.isLeaf:
            self.i = sigmoid(dot(self.Wi, x)+self.bi)
            self.o = sigmoid(dot(self.Wo, x)+self.bo)
            self.u = np.tanh(dot(self.Wu, x)+self.bu)
            node.hActs1 = np.multiply(self.i, self.u)
        else:
            for j in node.left:
                total += self.forwardProp(j)
            for j in node.right:
                total += self.forwardProp(j)
            si = add(dot(self.Wi, x),self.bi)
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                si += dot(self.Ui[idx], j.hActs2)
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                si += dot(self.Vi[idx], j.hActs2)
            self.i = sigmoid(si)

            su = add(dot(self.Wu, x),self.bu)
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                su += dot(self.Uu[idx], j.hActs2)
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                su += dot(self.Vu[idx], j.hActs2)
            self.u = np.tanh(su)

            so = add(dot(self.Wo, x),self.bo)
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                so += dot(self.Uo[idx], j.hActs2)
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                so += dot(self.Vo[idx], j.hActs2)
            self.o = sigmoid(so)

            temp = add(dot(self.Wf, x),self.bf)
            sl = np.zeros((self.middleDim), dtype='float32')
            sr = np.zeros((self.middleDim), dtype='float32')

            for j in range(self.paramDim):
                sl *= 0
                sl += temp
                for k in node.left:
                    idx2 = min(k.idx, self.paramDim-1)
                    sl += dot(self.Ul[j][idx2], k.hActs2)
                for k in node.right:
                    idx2 = min(k.idx, self.paramDim-1)
                    sl += dot(self.Vl[j][idx2], k.hActs2)
                self.l[j] = sigmoid(sl)

            for j in range(self.paramDim):
                sr *= 0
                sr += temp
                for k in node.left:
                    idx2 = min(k.idx, self.paramDim-1)
                    sr += dot(self.Ur[j][idx2], k.hActs2)
                for k in node.right:
                    idx2 = min(k.idx, self.paramDim-1)
                    sr += dot(self.Vr[j][idx2], k.hActs2)
                self.r[j] = sigmoid(sr)

            node.hActs1 = np.multiply(self.i, self.u)
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                node.hActs1 += np.multiply(self.l[idx], j.hActs1)
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                node.hActs1 += np.multiply(self.r[idx], j.hActs1)
        node.hActs2 = np.multiply(self.o, tanh(node.hActs1))
        return total + 1

    def backProp(self,node,error=None):

        dc_dsc = np.diag((1-node.hActs1**2).flatten())
        dh_dc = dotW(dc_dsc, np.diag(self.o.flatten()))
        # Inherited error
        if node.parent == None:
            error_at_h = error.astype('float32').squeeze()
            error_at_c = dot(dh_dc, error_at_h)
        if node.parent != None:
            [in_ho, in_hi, in_hu] = error[0:3]
            in_hl = error[3:3+self.paramDim]
            in_hr = error[3+self.paramDim:3+2*self.paramDim]
            in_cc = error[3+2*self.paramDim]
            if node in node.parent.left:
                idx = min(node.idx, self.paramDim-1)
                error_at_h = dot(self.Uo[idx].T, in_ho) + dot(self.Ui[idx].T, in_hi) + dot(self.Uu[idx].T, in_hu)
                for j in range(self.paramDim):
                    error_at_h += dot(self.Ul[j][idx].T, in_hl[j])
                    error_at_h += dot(self.Ur[j][idx].T, in_hr[j])
                error_at_c = dot(np.diag(self.l[idx].flatten()), in_cc) + dot(dh_dc, error_at_h)
            if node in node.parent.right:
                idx = min(node.idx, self.paramDim-1)
                error_at_h = dot(self.Vo[idx].T, in_ho) + dot(self.Vi[idx].T, in_hi) + dot(self.Vu[idx].T, in_hu)
                for j in range(self.paramDim):
                    error_at_h += dot(self.Vl[j][idx].T, in_hl[j])
                    error_at_h += dot(self.Vr[j][idx].T, in_hr[j])
                error_at_c = dot(np.diag(self.r[idx].flatten()), in_cc) + dot(dh_dc, error_at_h)
        # Error passed to children
        # o
        do_dso = np.diag(np.multiply(self.o, 1-self.o).flatten())
        dh_dso = dotW(do_dso, np.diag(np.tanh(node.hActs1).flatten()))
        # i
        di_dsi = np.diag(np.multiply(self.i, 1-self.i).flatten())
        dc_dsi = dotW(di_dsi, np.diag(self.u.flatten()))
        # u
        du_dsu = np.diag((1-self.u**2).flatten())
        dc_dsu = dotW(du_dsu, np.diag(self.i.flatten()))
        if not node.isLeaf:
            # l
            dl_dsl = []
            dc_dsl = []
            for j in range(self.paramDim):
                dc_dsl.append(np.zeros((self.middleDim, self.middleDim), dtype='float32'))
            for j in range(self.paramDim):
                dl_dsl.append(np.diag(np.multiply(self.l[j], 1-self.l[j]).flatten()))
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                dc_dsl[idx] += dotW(dl_dsl[idx], np.diag(j.hActs1.flatten()))
            # r
            dr_dsr = []
            dc_dsr = []
            for j in range(self.paramDim):
                dc_dsr.append(np.zeros((self.middleDim, self.middleDim), dtype='float32'))
            for j in range(self.paramDim):
                dr_dsr.append(np.diag(np.multiply(self.r[j], 1-self.r[j]).flatten()))
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                dc_dsr[idx] += dotW(dr_dsr[idx], np.diag(j.hActs1.flatten()))

            # Error out
            dJ_dso = dot(dh_dso, error_at_h)
            dJ_dsi = dot(dc_dsi, error_at_c)
            dJ_dsu = dot(dc_dsu, error_at_c)
            dJ_dsl = []
            dJ_dsr = []
            for j in range(self.paramDim):
                dJ_dsl.append(dot(dc_dsl[j], error_at_c))
                dJ_dsr.append(dot(dc_dsr[j], error_at_c))
            out_cc = error_at_c
            error_out = [dJ_dso, dJ_dsi, dJ_dsu]
            for j in range(self.paramDim):
                error_out.append(dJ_dsl[j])
            for j in range(self.paramDim):
                error_out.append(dJ_dsr[j])
            error_out.append(out_cc)
        # Parameter Gradients
        if not node.isLeaf:
            x = np.reshape(self.L[node.word,:], (self.wvecDim, 1))
            # Bias
            self.dbo += dJ_dso.flatten()
            self.dbi += dJ_dsi.flatten()
            self.dbu += dJ_dsu.flatten()
            for j in range(self.paramDim):
                self.dbf += dJ_dsl[j].flatten()
                self.dbf += dJ_dsr[j].flatten()
            # Us
            for j in node.left:
                idx = min(j.idx, self.paramDim-1)
                self.dUo[idx] += dotW(dJ_dso[:,None], j.hActs2[None,:])
                self.dUi[idx] += dotW(dJ_dsi[:,None], j.hActs2[None,:])
                self.dUu[idx] += dotW(dJ_dsu[:,None], j.hActs2[None,:])
                for k in range(self.paramDim):
                    self.dUl[k][idx] += dotW(dJ_dsl[k][:,None], j.hActs2[None,:])
                    self.dUr[k][idx] += dotW(dJ_dsr[k][:,None], j.hActs2[None,:])
            # Vs
            for j in node.right:
                idx = min(j.idx, self.paramDim-1)
                self.dVo[idx] += dotW(dJ_dso[:,None], j.hActs2[None,:])
                self.dVi[idx] += dotW(dJ_dsi[:,None], j.hActs2[None,:])
                self.dVu[idx] += dotW(dJ_dsu[:,None], j.hActs2[None,:])
                for k in range(self.paramDim):
                    self.dVl[k][idx] += dotW(dJ_dsl[k][:,None], j.hActs2[None,:])
                    self.dVr[k][idx] += dotW(dJ_dsr[k][:,None], j.hActs2[None,:])
            # Ws
            self.dWo += dotW(dJ_dso[:,None], x.T)
            self.dWu += dotW(dJ_dsu[:,None], x.T)
            self.dWi += dotW(dJ_dsi[:,None], x.T)
            for j in range(self.paramDim):
                self.dWf += dotW(dJ_dsl[j][:,None], x.T)
                self.dWf += dotW(dJ_dsr[j][:,None], x.T)
            # L
            temp = dot(self.Wo.T, dJ_dso).flatten() + dot(self.Wi.T, dJ_dsi).flatten() + dot(self.Wu.T, dJ_dsu).flatten()
            for j in range(self.paramDim):
                temp += dot(self.Wf.T, dJ_dsl[j]).flatten()
                temp += dot(self.Wf.T, dJ_dsr[j]).flatten()
            self.dL[node.word] = temp

            # Recursion
            for j in node.left:
                self.backProp(j, error_out)
            for j in node.right:
                self.backProp(j, error_out)
        else:
            x = np.reshape(self.L[node.word,:], (self.wvecDim, 1))
            dJ_dso = dot(dh_dso, error_at_h)
            dJ_dsi = dot(dc_dsi, error_at_c)
            dJ_dsu = dot(dc_dsu, error_at_c)
            # Bias
            self.dbo += dJ_dso.flatten()
            self.dbi += dJ_dsi.flatten()
            self.dbu += dJ_dsu.flatten()
            # Ws
            self.dWo += dotW(dJ_dso[:,None], x.T)
            self.dWi += dotW(dJ_dsi[:,None], x.T)
            self.dWu += dotW(dJ_dsu[:,None], x.T)
            # L
            self.dL[node.word] = dot(self.Wo.T, dJ_dso).flatten() + dot(self.Wi.T, dJ_dsi).flatten() + dot(self.Wu.T, dJ_dsu).flatten()
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[j,:] += scale*dL[j]

    def toFile(self):
        return self.stack

    def fromFile(self,stack):
        self.stack = stack

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dW..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None]
            err2 = 0.0
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    err2+=err
                    count+=1
            print W.shape, err2/count
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)
        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                err2+=err
                count+=1
        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)


























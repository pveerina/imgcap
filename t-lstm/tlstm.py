import numpy as np
import collections
import pdb
np.seterr(under='warn')

# This is a Tree-structured LSTM
# You must update the forward and backward propogation functions of this file.

# You can run this file via 'python rnn2deep.py' to perform a gradient check

# tip: insert pdb.set_trace() in places where you are unsure whats going on

def softmax(x):
	if x.ndim == 1:
		x = np.exp(x-np.max(x))
		x = x/np.sum(x)
	else:
		n = x.shape[0]
		x = np.exp(x-np.reshape(np.max(x, axis=1), [n, 1]))
		x = x/np.reshape(np.sum(x, axis=1), [n, 1])
	return x

def sigmoid(x):
	return 1/(1+np.exp(-x))

def make_onehot(index, length):
	y = np.zeros(length)
	y[index] = 1
	return y

class TLSTM:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Bias Terms
        self.bs = np.zeros((self.outputDim))
	self.bf = np.zeros((self.middleDim))
	self.bi = np.zeros((self.middleDim))
	self.bo = np.zeros((self.middleDim))
	self.bu = np.zeros((self.middleDim))
	
	# Input Weights
	self.Ws = 0.1*np.random.randn(self.outputDim, self.middleDim)
	self.Wu = 0.1*np.random.randn(self.middleDim, self.wvecDim)
	self.Wo = 0.1*np.random.randn(self.middleDim, self.wvecDim)
	self.Wi = 0.1*np.random.randn(self.middleDim, self.wvecDim)
	self.Wf = 0.1*np.random.randn(self.middleDim, self.wvecDim)
	
	# Left Hidden Weights
	self.Ui = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Ul = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Ur = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Uo = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Uu = 0.1*np.random.randn(self.middleDim, self.middleDim)
	
	# Right Hidden Weights
	self.Vi = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Vl = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Vr = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Vo = 0.1*np.random.randn(self.middleDim, self.middleDim)
	self.Vu = 0.1*np.random.randn(self.middleDim, self.middleDim)
	
        self.stack = [self.L, self.Ws, self.bs, self.Wo, self.Uo, self.Vo, self.bo, self.Wi, self.Ui, self.Vi, self.bi, self.Wu, self.Uu, self.Vu, self.bu, self.Wf, self.Ul, self.Vl, self.Ur, self.Vr, self.bf]

        # Gradients
        self.dbs = np.empty((self.outputDim))
	self.dbi = np.empty((self.middleDim))
	self.dbo = np.empty((self.middleDim))
	self.dbu = np.empty((self.middleDim))
	self.dbf = np.empty((self.middleDim))
	
	self.dWs = np.empty(self.Ws.shape)
	self.dWf = np.empty(self.Wf.shape)
	self.dWi = np.empty(self.Wi.shape)
	self.dWo = np.empty(self.Wo.shape)
	self.dWu = np.empty(self.Wu.shape)

	self.dUl = np.empty(self.Ul.shape)
	self.dUr = np.empty(self.Ur.shape)
	self.dUi = np.empty(self.Ui.shape)
	self.dUo = np.empty(self.Uo.shape)
	self.dUu = np.empty(self.Uu.shape)
        
	self.dVl = np.empty(self.Vl.shape)
	self.dVr = np.empty(self.Vr.shape)
	self.dVi = np.empty(self.Vi.shape)
	self.dVo = np.empty(self.Vo.shape)
	self.dVu = np.empty(self.Vu.shape)
        
    def costAndGrad(self,mbdata,test=False): 
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
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.Ws, self.bs, self.Wo, self.Uo, self.Vo, self.bo, self.Wi, self.Ui, self.Vi, self.bi, self.Wu, self.Uu, self.Vu, self.bu, self.Wf, self.Ul, self.Vl, self.Ur, self.Vr, self.bf = self.stack
        # Zero gradients
	self.dbs[:] = 0
	self.dbi[:] = 0
	self.dbo[:] = 0
	self.dbu[:] = 0
	self.dbf[:] = 0

	self.dWs[:] = 0
	self.dWu[:] = 0
	self.dWi[:] = 0
	self.dWo[:] = 0
	self.dWf[:] = 0

	self.dUl[:] = 0
	self.dUr[:] = 0
	self.dUi[:] = 0
	self.dUo[:] = 0
	self.dUu[:] = 0

	self.dVl[:] = 0
	self.dVr[:] = 0
	self.dVi[:] = 0
	self.dVo[:] = 0
	self.dVu[:] = 0

        self.dL = collections.defaultdict(self.defaultVec)

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
            cost += c
            total += tot
            
        if test:
            return (1./len(mbdata))*cost,correct, guess, total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.Ws**2)
        cost += (self.rho/2)*np.sum(self.Wf**2)
        cost += (self.rho/2)*np.sum(self.Wi**2)
        cost += (self.rho/2)*np.sum(self.Wo**2)
        cost += (self.rho/2)*np.sum(self.Wu**2)

        cost += (self.rho/2)*np.sum(self.Ul**2)
        cost += (self.rho/2)*np.sum(self.Ur**2)
        cost += (self.rho/2)*np.sum(self.Ui**2)
        cost += (self.rho/2)*np.sum(self.Uo**2)
        cost += (self.rho/2)*np.sum(self.Uu**2)
        
        cost += (self.rho/2)*np.sum(self.Vl**2)
        cost += (self.rho/2)*np.sum(self.Vr**2)
        cost += (self.rho/2)*np.sum(self.Vi**2)
        cost += (self.rho/2)*np.sum(self.Vo**2)
        cost += (self.rho/2)*np.sum(self.Vu**2)
        
	# UPDATED TO THIS POINT #
	return scale*cost, [self.dL, scale*(self.dWs + self.Ws*self.rho), scale*self.dbs, scale*(self.dWo + self.Wo*self.rho), scale*(self.dUo + self.Uo*self.rho), scale*(self.dVo + self.Vo*self.rho), scale*self.dbo, scale*(self.dWi + self.Wi*self.rho), scale*(self.dUi + self.Ui*self.rho), scale*(self.dVi + self.Vi*self.rho), scale*self.dbi, scale*(self.dWu + self.Wu*self.rho), scale*(self.dUu + self.Uu*self.rho), scale*(self.dVu + self.Vu*self.rho), scale*self.dbu, scale*(self.dWf + self.Wf*self.rho), scale*(self.dUl + self.Ul*self.rho), scale*(self.dVl + self.Vl*self.rho), scale*(self.dUr + self.Ur*self.rho), scale*(self.dVr + self.Vr*self.rho), scale*self.dbf]

    def forwardProp(self,node, correct=[], guess=[]):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
	if node.isLeaf:
		x = np.reshape(self.L[:, node.word], (self.wvecDim, 1))
		self.i = sigmoid(np.dot(self.Wi, x)+np.reshape(self.bi, (self.middleDim, 1)))
		self.o = sigmoid(np.dot(self.Wo, x)+np.reshape(self.bo, (self.middleDim, 1)))
		self.u = np.tanh(np.dot(self.Wu, x)+np.reshape(self.bu, (self.middleDim, 1)))
		#self.i = np.maximum(np.dot(self.Wi, x)+np.reshape(self.bi, (self.middleDim, 1)), 0)
		#self.o = np.maximum(np.dot(self.Wo, x)+np.reshape(self.bo, (self.middleDim, 1)), 0)
		#self.u = np.maximum(np.dot(self.Wu, x)+np.reshape(self.bu, (self.middleDim, 1)), 0)
		node.hActs1 = np.multiply(self.i, self.u)
	else:
		cost_left, total_left = self.forwardProp(node.left, correct, guess)
		cost_right, total_right = self.forwardProp(node.right, correct, guess)
		self.i = sigmoid(np.dot(self.Ui, node.left.hActs2)+np.dot(self.Vi, node.right.hActs2)+np.reshape(self.bi, (self.middleDim, 1)))
		self.u = np.tanh(np.dot(self.Uu, node.left.hActs2)+np.dot(self.Vu, node.right.hActs2)+np.reshape(self.bu, (self.middleDim, 1)))
		self.o = sigmoid(np.dot(self.Uo, node.left.hActs2)+np.dot(self.Vo, node.right.hActs2)+np.reshape(self.bo, (self.middleDim, 1)))
		self.l = sigmoid(np.dot(self.Ul, node.left.hActs2)+np.dot(self.Vl, node.right.hActs2)+np.reshape(self.bf, (self.middleDim, 1)))
		self.r = sigmoid(np.dot(self.Ur, node.left.hActs2)+np.dot(self.Vr, node.right.hActs2)+np.reshape(self.bf, (self.middleDim, 1)))
		#self.i = np.maximum(np.dot(self.Ui, node.left.hActs2)+np.dot(self.Vi, node.right.hActs2)+np.reshape(self.bi, (self.middleDim, 1)), 0)
		#self.u = np.maximum(np.dot(self.Uu, node.left.hActs2)+np.dot(self.Vu, node.right.hActs2)+np.reshape(self.bu, (self.middleDim, 1)), 0)
		#self.o = np.maximum(np.dot(self.Uo, node.left.hActs2)+np.dot(self.Vo, node.right.hActs2)+np.reshape(self.bo, (self.middleDim, 1)), 0)
		#self.l = np.maximum(np.dot(self.Ul, node.left.hActs2)+np.dot(self.Vl, node.right.hActs2)+np.reshape(self.bf, (self.middleDim, 1)), 0)
		#self.r = np.maximum(np.dot(self.Ur, node.left.hActs2)+np.dot(self.Vr, node.right.hActs2)+np.reshape(self.bf, (self.middleDim, 1)), 0)
		node.hActs1 = np.multiply(self.i, self.u)+np.multiply(self.l, node.left.hActs1)+np.multiply(self.r, node.right.hActs1)
	node.hActs2 = np.multiply(self.o, np.tanh(node.hActs1))
	node.probs = softmax((np.dot(self.Ws, node.hActs2)+np.reshape(self.bs, (self.outputDim, 1))).flatten())
	guess.append(np.argmax(node.probs))
	cost += -np.log(node.probs[node.label])
	correct.append(node.label)
	if not node.isLeaf:
		cost += cost_left + cost_right
		total += total_left + total_right
	node.fprop = True
        return cost, total + 1

    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False

        # this is exactly the same setup as backProp in rnn.py
	# theta
	dJ_dtheta = np.reshape(node.probs-make_onehot(node.label, self.outputDim), (self.outputDim, 1))
	# h
	dJ_dh = np.dot(self.Ws.T, dJ_dtheta)
	# c
	dc_dsc = np.diag((1-node.hActs1**2).flatten())
	dh_dc = np.dot(dc_dsc, np.diag(self.o.flatten()))
	dJ_dc = np.dot(dh_dc, dJ_dh)
	# Error produced within cell
	error_at_h = dJ_dh
	error_at_c = dJ_dc
	# Inherited error
	if node.parent != None:
		[in_ho, in_hi, in_hu, in_hl, in_hr, in_cc] = error
		if node == node.parent.left:
			error_at_h += np.dot(self.Uo.T, in_ho) + np.dot(self.Ui.T, in_hi) + np.dot(self.Uu.T, in_hu) + np.dot(self.Ul.T, in_hl) + np.dot(self.Ur.T, in_hr)
			error_at_c += np.dot(np.diag(self.l.flatten()), in_cc) + np.dot(dh_dc, error_at_h)
		if node == node.parent.right:
			error_at_h += np.dot(self.Vo.T, in_ho) + np.dot(self.Vi.T, in_hi) + np.dot(self.Vu.T, in_hu) + np.dot(self.Vl.T, in_hl) + np.dot(self.Vr.T, in_hr)
			error_at_c += np.dot(np.diag(self.r.flatten()), in_cc) + np.dot(dh_dc, error_at_h)
	# Error passed to children
	# o
	do_dso = np.diag(np.multiply(self.o, 1-self.o).flatten())
	#do_dso = np.diag(np.maximum(np.sign(self.o), 0).flatten())
	dh_dso = np.dot(do_dso, np.diag(np.tanh(node.hActs1).flatten()))
	#out_ho = np.dot(dh_dso, dJ_dh)
	# i
	di_dsi = np.diag(np.multiply(self.i, 1-self.i).flatten())
	#di_dsi = np.diag(np.maximum(np.sign(self.i), 0).flatten())
	dc_dsi = np.dot(di_dsi, np.diag(self.u.flatten()))
	#out_hi = np.dot(dc_dsi, dJ_dc)
	# u
	du_dsu = np.diag((1-self.u**2).flatten())
	dc_dsu = np.dot(du_dsu, np.diag(self.i.flatten()))
	#out_hu = np.dot(dc_dsu, dJ_dc)
	if not node.isLeaf:
		# l
		dl_dsl = np.diag(np.multiply(self.l, 1-self.l).flatten())
		#dl_dsl = np.diag(np.maximum(np.sign(self.l), 0).flatten())
		dc_dsl = np.dot(dl_dsl, np.diag(node.left.hActs1.flatten()))
		#out_hl = np.dot(dc_dsl, dJ_dc)
		# r
		dr_dsr = np.diag(np.multiply(self.r, 1-self.r).flatten())
		#dr_dsr = np.diag(np.maximum(np.sign(self.r), 0).flatten())
		dc_dsr = np.dot(dr_dsr, np.diag(node.right.hActs1.flatten()))
		#out_hr = np.dot(dc_dsr, dJ_dc)
		# c
		#out_cc = dJ_dc
		# Error out
		#out_ho = np.dot(dh_dso, error_at_h)
		#out_hi = np.dot(dc_dsi, error_at_c)
		#out_hu = np.dot(dc_dsu, error_at_c)
		#out_hl = np.dot(dc_dsl, error_at_c)
		#out_hr = np.dot(dc_dsr, error_at_c)
		#out_cc = error_at_c
		#error_out = [out_ho, out_hi, out_hu, out_hl, out_hr, out_cc]
		dJ_dso = np.dot(dh_dso, error_at_h)
		dJ_dsi = np.dot(dc_dsi, error_at_c)
		dJ_dsu = np.dot(dc_dsu, error_at_c)
		dJ_dsl = np.dot(dc_dsl, error_at_c)
		dJ_dsr = np.dot(dc_dsr, error_at_c)
		out_cc = error_at_c
		error_out = [dJ_dso, dJ_dsi, dJ_dsu, dJ_dsl, dJ_dsr, out_cc]
	# Parameter Gradients
	if not node.isLeaf:
		#dJ_dso = np.dot(dh_dso, error_at_h)
		#dJ_dsi = np.dot(dc_dsi, error_at_c)
		#dJ_dsu = np.dot(dc_dsu, error_at_c)
		#dJ_dsl = np.dot(dc_dsl, error_at_c)
		#dJ_dsr = np.dot(dc_dsr, error_at_c)
		# Bias
		self.dbs += dJ_dtheta.flatten()
		self.dbo += dJ_dso.flatten()
		self.dbi += dJ_dsi.flatten()
		self.dbu += dJ_dsu.flatten()
		self.dbf += dJ_dsl.flatten() + dJ_dsr.flatten()
		# Ws
		self.dWs += np.dot(dJ_dtheta, node.hActs2.T)
		# Us
		self.dUo += np.dot(dJ_dso, node.left.hActs2.T)
		self.dUi += np.dot(dJ_dsi, node.left.hActs2.T)
		self.dUu += np.dot(dJ_dsu, node.left.hActs2.T)
		self.dUl += np.dot(dJ_dsl, node.left.hActs2.T)
		self.dUr += np.dot(dJ_dsr, node.left.hActs2.T)
		# Vs
		self.dVo += np.dot(dJ_dso, node.right.hActs2.T)
		self.dVi += np.dot(dJ_dsi, node.right.hActs2.T)
		self.dVu += np.dot(dJ_dsu, node.right.hActs2.T)
		self.dVl += np.dot(dJ_dsl, node.right.hActs2.T)
		self.dVr += np.dot(dJ_dsr, node.right.hActs2.T)
		# Recursion
		self.backProp(node.left, error_out)
		self.backProp(node.right, error_out)
	else:
		x = np.reshape(self.L[:, node.word], (self.wvecDim, 1))
		dJ_dso = np.dot(dh_dso, error_at_h)
		dJ_dsi = np.dot(dc_dsi, error_at_c)
		dJ_dsu = np.dot(dc_dsu, error_at_c)
		# Bias
		self.dbs += dJ_dtheta.flatten()
		self.dbo += dJ_dso.flatten()
		self.dbi += dJ_dsi.flatten()
		self.dbu += dJ_dsu.flatten()
		# Ws
		self.dWs += np.dot(dJ_dtheta, node.hActs2.T)
		self.dWo += np.dot(dJ_dso, x.T)
		self.dWi += np.dot(dJ_dsi, x.T)
		self.dWu += np.dot(dJ_dsu, x.T)
		# L
		self.dL[node.word] = np.dot(self.Wo.T, dJ_dso).flatten() + np.dot(self.Wi.T, dJ_dsi).flatten() + np.dot(self.Wu.T, dJ_dsu).flatten()
    
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
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dW..."
        for W,dW in zip(self.stack[1:],grad[1:]):
	    W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
	    print W.shape, err1/count
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

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    middleDim = 10
    outputDim = 5

    rnn = TLSTM(wvecDim,middleDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)







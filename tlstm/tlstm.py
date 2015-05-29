import numpy as np
import collections
import pdb
np.seterr(under='warn')
import theano as th
import theano.tensor as t

# This is a Tree-structured LSTM
# You must update the forward and backward propogation functions of this file.

# You can run this file via 'python tlstm.py' to perform a gradient check

# tip: insert pdb.set_trace() in places where you are unsure whats going on

# Softmax
x = t.dvector('x')
y = t.exp(x-t.max(x))
y = y/t.sum(y)
softmax = th.function([x], y)

# Sigmoid
x = t.dmatrix('x')
y = 1/(1+t.exp(-x))
sigmoid = th.function([x], y)

def make_onehot(index, length):
	y = np.zeros(length)
	y[index] = 1
	return y

class TLSTM:

    def __init__(self,wvecDim, middleDim, paramDim, numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.middleDim = middleDim
	self.paramDim = paramDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)
        
	# Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Bias Terms
	self.bf = np.zeros((self.middleDim))
	self.bi = np.zeros((self.middleDim))
	self.bo = np.zeros((self.middleDim))
	self.bu = np.zeros((self.middleDim))
	
	# Input Weights
	self.Wu = 0.01*np.random.randn(self.middleDim, self.wvecDim)
	self.Wo = 0.01*np.random.randn(self.middleDim, self.wvecDim)
	self.Wi = 0.01*np.random.randn(self.middleDim, self.wvecDim)
	self.Wf = 0.01*np.random.randn(self.middleDim, self.wvecDim)
	
	# Left Hidden Weights
	self.Ui = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	self.Ul = [[0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.Ur = [[0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.Uo = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	self.Uu = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	
	# Right Hidden Weights
	self.Vi = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	self.Vl = [[0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.Vr = [[0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.Vo = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	self.Vu = [0.01*np.random.randn(self.middleDim, self.middleDim) for j in range(self.paramDim)]
	
        self.stack = [self.L]

	self.stack.append(self.Wo)
	for j in range(self.paramDim):
		self.stack.append(self.Uo[j])
	for j in range(self.paramDim):
		self.stack.append(self.Vo[j])
	self.stack.append(self.bo)

	self.stack.append(self.Wi)
	for j in range(self.paramDim):
		self.stack.append(self.Ui[j])
	for j in range(self.paramDim):
		self.stack.append(self.Vi[j])
	self.stack.append(self.bi)

	self.stack.append(self.Wu)
	for j in range(self.paramDim):
		self.stack.append(self.Uu[j])
	for j in range(self.paramDim):
		self.stack.append(self.Vu[j])
	self.stack.append(self.bu)
	
	self.stack.append(self.Wf)
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.stack.append(self.Ul[j][k])
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.stack.append(self.Vl[j][k])
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.stack.append(self.Ur[j][k])
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.stack.append(self.Vr[j][k])
	self.stack.append(self.bf)

        # Gradients
	self.dbi = np.empty((self.middleDim))
	self.dbo = np.empty((self.middleDim))
	self.dbu = np.empty((self.middleDim))
	self.dbf = np.empty((self.middleDim))
	
	self.dWf = np.empty(self.Wf.shape)
	self.dWi = np.empty(self.Wi.shape)
	self.dWo = np.empty(self.Wo.shape)
	self.dWu = np.empty(self.Wu.shape)

	self.dUl = [[np.empty(self.Ul[0][0].shape) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.dUr = [[np.empty(self.Ur[0][0].shape) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.dUi = [np.empty(self.Ui[0].shape) for j in range(self.paramDim)]
	self.dUo = [np.empty(self.Uo[0].shape) for j in range(self.paramDim)]
	self.dUu = [np.empty(self.Uu[0].shape) for j in range(self.paramDim)]
        
	self.dVl = [[np.empty(self.Vl[0][0].shape) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.dVr = [[np.empty(self.Vr[0][0].shape) for j in range(self.paramDim)] for k in range(self.paramDim)]
	self.dVi = [np.empty(self.Vi[0].shape) for j in range(self.paramDim)]
	self.dVo = [np.empty(self.Vo[0].shape) for j in range(self.paramDim)]
	self.dVu = [np.empty(self.Vu[0].shape) for j in range(self.paramDim)]
        
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

        self.L = self.stack[0]
	
	self.Wo = self.stack[1]
	self.Uo = self.stack[2:2+self.paramDim]
	self.Vo = self.stack[2+self.paramDim:2+2*self.paramDim]
	self.bo = self.stack[2+2*self.paramDim]

	self.Wi = self.stack[3+2*self.paramDim]
	self.Ui = self.stack[4+2*self.paramDim:4+3*self.paramDim]
	self.Vi = self.stack[4+3*self.paramDim:4+4*self.paramDim]
	self.bi = self.stack[4+4*self.paramDim]

	self.Wu = self.stack[5+4*self.paramDim]
	self.Uu = self.stack[6+4*self.paramDim:6+5*self.paramDim]
	self.Vu = self.stack[6+5*self.paramDim:6+6*self.paramDim]
	self.bu = self.stack[6+6*self.paramDim]

	self.Wf = self.stack[7+6*self.paramDim]
	self.Ul = [self.stack[8+(6+j)*self.paramDim:8+(7+j)*self.paramDim] for j in range(self.paramDim)]
	self.Vl = [self.stack[8+(7+self.paramDim-1+j)*self.paramDim:8+(7+self.paramDim+j)*self.paramDim] for j in range(self.paramDim)]
	self.Ur = [self.stack[8+(7+2*self.paramDim-1+j)*self.paramDim:8+(7+2*self.paramDim+j)*self.paramDim] for j in range(self.paramDim)]
	self.Vr = [self.stack[8+(7+3*self.paramDim-1+j)*self.paramDim:8+(7+3*self.paramDim+j)*self.paramDim] for j in range(self.paramDim)]
	self.bf = self.stack[8+(7+4*self.paramDim-1)*self.paramDim]

	assert len(self.stack) == 9+(6+4*self.paramDim)*self.paramDim

        # Zero gradients
	self.dbi[:] = 0
	self.dbo[:] = 0
	self.dbu[:] = 0
	self.dbf[:] = 0

	self.dWu[:] = 0
	self.dWi[:] = 0
	self.dWo[:] = 0
	self.dWf[:] = 0

	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.dUl[j][k][:] = 0
			self.dUr[j][k][:] = 0
		self.dUi[j][:] = 0
		self.dUo[j][:] = 0
		self.dUu[j][:] = 0

	for j in range(self.paramDim):
		for k in range(self.paramDim):
			self.dVl[j][k][:] = 0
			self.dVr[j][k][:] = 0
		self.dVi[j][:] = 0
		self.dVo[j][:] = 0
		self.dVu[j][:] = 0

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
        cost += (self.rho/2)*np.sum(self.Wf**2)
        cost += (self.rho/2)*np.sum(self.Wi**2)
        cost += (self.rho/2)*np.sum(self.Wo**2)
        cost += (self.rho/2)*np.sum(self.Wu**2)

	for j in range(self.paramDim):
		for k in range(self.paramDim):
			cost += (self.rho/2)*np.sum(self.Ul[j][k]**2)
			cost += (self.rho/2)*np.sum(self.Ur[j][k]**2)
		cost += (self.rho/2)*np.sum(self.Ui[j]**2)
		cost += (self.rho/2)*np.sum(self.Uo[j]**2)
		cost += (self.rho/2)*np.sum(self.Uu[j]**2)
        
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			cost += (self.rho/2)*np.sum(self.Vl[j][k]**2)
			cost += (self.rho/2)*np.sum(self.Vr[j][k]**2)
		cost += (self.rho/2)*np.sum(self.Vi[j]**2)
		cost += (self.rho/2)*np.sum(self.Vo[j]**2)
		cost += (self.rho/2)*np.sum(self.Vu[j]**2)
        
	grad_stack = [self.dL]

	grad_stack.append(scale*(self.dWo + self.Wo*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dUo[j] + self.Uo[j]*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dVo[j] + self.Vo[j]*self.rho))
	grad_stack.append(scale*self.dbo)
	
	grad_stack.append(scale*(self.dWi + self.Wi*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dUi[j] + self.Ui[j]*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dVi[j] + self.Vi[j]*self.rho))
	grad_stack.append(scale*self.dbi)
	
	grad_stack.append(scale*(self.dWu + self.Wu*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dUu[j] + self.Uu[j]*self.rho))
	for j in range(self.paramDim):
		grad_stack.append(scale*(self.dVu[j] + self.Vu[j]*self.rho))
	grad_stack.append(scale*self.dbu)
	
	grad_stack.append(scale*(self.dWf + self.Wf*self.rho))
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			grad_stack.append(scale*(self.dUl[j][k] + self.Ul[j][k]*self.rho))
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			grad_stack.append(scale*(self.dVl[j][k] + self.Vl[j][k]*self.rho))
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			grad_stack.append(scale*(self.dUr[j][k] + self.Ur[j][k]*self.rho))
	for j in range(self.paramDim):
		for k in range(self.paramDim):
			grad_stack.append(scale*(self.dVr[j][k] + self.Vr[j][k]*self.rho))
	grad_stack.append(scale*self.dbf)

	return scale*cost, grad_stack 

    def forwardProp(self,node, correct=[], guess=[]):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
	x = np.reshape(self.L[:, node.word], (self.wvecDim, 1))
	if node.isLeaf:
		self.i = sigmoid(np.dot(self.Wi, x)+np.reshape(self.bi, (self.middleDim, 1)))
		self.o = sigmoid(np.dot(self.Wo, x)+np.reshape(self.bo, (self.middleDim, 1)))
		self.u = np.tanh(np.dot(self.Wu, x)+np.reshape(self.bu, (self.middleDim, 1)))
		node.hActs1 = np.multiply(self.i, self.u)
	else:
		cost_left = 0
		total_left = 0
		for j in node.left:
			cost_temp, total_temp = self.forwardProp(j, correct, guess)
			cost_left += cost_temp
			total_left += total_temp
		cost_right = 0
		total_right = 0
		for j in node.right:
			cost_temp, total_temp = self.forwardProp(j, correct, guess)
			cost_right += cost_temp
			total_right += total_temp
		
		si = np.dot(self.Wi, x)+np.reshape(self.bi, (self.middleDim, 1))
		for j in node.left:
			idx = min(j.idx, self.paramDim-1)
			si += np.dot(self.Ui[idx], j.hActs2)
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			si += np.dot(self.Vi[idx], j.hActs2)
		self.i = sigmoid(si)
		
		su = np.dot(self.Wu, x)+np.reshape(self.bu, (self.middleDim, 1))
		for j in node.left:
			idx = min(j.idx, self.paramDim-1)
			su += np.dot(self.Uu[idx], j.hActs2)
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			su += np.dot(self.Vu[idx], j.hActs2)
		self.u = np.tanh(su)
		
		so = np.dot(self.Wo, x)+np.reshape(self.bo, (self.middleDim, 1))
		for j in node.left:
			idx = min(j.idx, self.paramDim-1)
			so += np.dot(self.Uo[idx], j.hActs2)
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			so += np.dot(self.Vo[idx], j.hActs2)
		self.o = sigmoid(so)
		
		self.l = []
		sl = []
		for j in range(self.paramDim):
			self.l.append(np.zeros((self.middleDim, 1)))
			sl.append(np.zeros((self.middleDim, 1)))
		temp = np.dot(self.Wf, x) + np.reshape(self.bf, (self.middleDim, 1))
		for j in range(self.paramDim):
			idx1 = j
			sl[idx1] += temp
			for k in node.left:
				idx2 = min(k.idx, self.paramDim-1)
				sl[idx1] += np.dot(self.Ul[idx1][idx2], k.hActs2)
			for k in node.right:
				idx2 = min(k.idx, self.paramDim-1)
				sl[idx1] += np.dot(self.Vl[idx1][idx2], k.hActs2)
		for j in range(self.paramDim):
			self.l[j] = sigmoid(sl[j])
				
		self.r = []
		sr = []
		for j in range(self.paramDim):
			self.r.append(np.zeros((self.middleDim, 1)))
			sr.append(np.zeros((self.middleDim, 1)))
		for j in range(self.paramDim):
			idx1 = j
			sl[idx1] += temp
			for k in node.left:
				idx2 = min(k.idx, self.paramDim-1)
				sr[idx1] += np.dot(self.Ur[idx1][idx2], k.hActs2)
			for k in node.right:
				idx2 = min(k.idx, self.paramDim-1)
				sr[idx1] += np.dot(self.Vr[idx1][idx2], k.hActs2)
		for j in range(self.paramDim):
			self.r[j] = sigmoid(sr[j])
				
		node.hActs1 = np.multiply(self.i, self.u)
		for j in node.left:
			idx = min(j.idx, self.paramDim-1)
			node.hActs1 += np.multiply(self.l[idx], j.hActs1)
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			node.hActs1 += np.multiply(self.r[idx], j.hActs1)
	node.hActs2 = np.multiply(self.o, np.tanh(node.hActs1))
	node.probs = softmax(node.hActs2.flatten())
	guess.append(np.argmax(node.probs))
	node.label = np.mod(node.word, middleDim)
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
	dJ_dtheta = np.reshape(node.probs-make_onehot(node.label, self.middleDim), (self.middleDim, 1))
	# h
	dJ_dh = dJ_dtheta
	# c
	dc_dsc = np.diag((1-node.hActs1**2).flatten())
	dh_dc = np.dot(dc_dsc, np.diag(self.o.flatten()))
	dJ_dc = np.dot(dh_dc, dJ_dh)
	# Error produced within cell
	error_at_h = dJ_dh
	error_at_c = dJ_dc
	# Inherited error
	if node.parent != None:
		[in_ho, in_hi, in_hu] = error[0:3]
		in_hl = error[3:3+self.paramDim]
		in_hr = error[3+self.paramDim:3+2*self.paramDim]
		in_cc = error[3+2*self.paramDim]
		if node == node.parent.left:
			idx = min(node.idx, self.paramDim-1)
			error_at_h += np.dot(self.Uo[idx].T, in_ho) + np.dot(self.Ui[idx].T, in_hi) + np.dot(self.Uu[idx].T, in_hu)
			for j in range(self.paramDim):
				error_at_h += np.dot(self.Ul[j][idx].T, in_hl[j])
				error_at_h += np.dot(self.Ur[j][idx].T, in_hr[j])
			error_at_c += np.dot(np.diag(self.l[idx].flatten()), in_cc) + np.dot(dh_dc, error_at_h)
		if node == node.parent.right:
			idx = min(node.idx, self.paramDim-1)
			error_at_h += np.dot(self.Vo[idx].T, in_ho) + np.dot(self.Vi[idx].T, in_hi) + np.dot(self.Vu[idx].T, in_hu)
			for j in range(self.paramDim):
				error_at_h += np.dot(self.Vl[j][idx].T, in_hl[j])
				error_at_h += np.dot(self.Vr[j][idx].T, in_hr[j])
			error_at_c += np.dot(np.diag(self.r[idx].flatten()), in_cc) + np.dot(dh_dc, error_at_h)
	# Error passed to children
	# o
	do_dso = np.diag(np.multiply(self.o, 1-self.o).flatten())
	dh_dso = np.dot(do_dso, np.diag(np.tanh(node.hActs1).flatten()))
	# i
	di_dsi = np.diag(np.multiply(self.i, 1-self.i).flatten())
	dc_dsi = np.dot(di_dsi, np.diag(self.u.flatten()))
	# u
	du_dsu = np.diag((1-self.u**2).flatten())
	dc_dsu = np.dot(du_dsu, np.diag(self.i.flatten()))
	if not node.isLeaf:
		
		# l
		dl_dsl = []
		dc_dsl = []
		for j in range(self.paramDim):
			dc_dsl.append(np.zeros((self.middleDim, self.middleDim)))
		for j in range(self.paramDim):
			dl_dsl.append(np.diag(np.multiply(self.l[j], 1-self.l[j]).flatten()))
		for j in node.left:
			idx = min(j.idx, self.paramDim-1)
			dc_dsl[idx] += np.dot(dl_dsl[idx], np.diag(j.hActs1.flatten()))
		# r
		dr_dsr = []
		dc_dsr = []
		for j in range(self.paramDim):
			dc_dsr.append(np.zeros((self.middleDim, self.middleDim)))
		for j in range(self.paramDim):
			dr_dsr.append(np.diag(np.multiply(self.r[j], 1-self.r[j]).flatten()))
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			dc_dsr[idx] += np.dot(dr_dsr[idx], np.diag(j.hActs1.flatten()))
		
		# Error out
		dJ_dso = np.dot(dh_dso, error_at_h)
		dJ_dsi = np.dot(dc_dsi, error_at_c)
		dJ_dsu = np.dot(dc_dsu, error_at_c)
		dJ_dsl = []
		dJ_dsr = []
		for j in range(self.paramDim):
			dJ_dsl.append(np.dot(dc_dsl[j], error_at_c))
			dJ_dsr.append(np.dot(dc_dsr[j], error_at_c))
		out_cc = error_at_c
		error_out = [dJ_dso, dJ_dsi, dJ_dsu]
		for j in range(self.paramDim):
			error_out.append(dJ_dsl[j])
		for j in range(self.paramDim):
			error_out.append(dJ_dsr[j])
		error_out.append(out_cc)
	# Parameter Gradients
	if not node.isLeaf:
		x = np.reshape(self.L[:, node.word], (self.wvecDim, 1))
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
			self.dUo[idx] += np.dot(dJ_dso, j.hActs2.T)
			self.dUi[idx] += np.dot(dJ_dsi, j.hActs2.T)
			self.dUu[idx] += np.dot(dJ_dsu, j.hActs2.T)
			for k in range(self.paramDim):
				self.dUl[k][idx] += np.dot(dJ_dsl[k], j.hActs2.T)
				self.dUr[k][idx] += np.dot(dJ_dsr[k], j.hActs2.T)
		# Vs
		for j in node.right:
			idx = min(j.idx, self.paramDim-1)
			self.dVo[idx] += np.dot(dJ_dso, j.hActs2.T)
			self.dVi[idx] += np.dot(dJ_dsi, j.hActs2.T)
			self.dVu[idx] += np.dot(dJ_dsu, j.hActs2.T)
			for k in range(self.paramDim):
				self.dVl[k][idx] += np.dot(dJ_dsl[k], j.hActs2.T)
				self.dVr[k][idx] += np.dot(dJ_dsr[k], j.hActs2.T)
		# Ws
		self.dWo += np.dot(dJ_dso, x.T) 
		self.dWu += np.dot(dJ_dsu, x.T) 
		self.dWi += np.dot(dJ_dsi, x.T)
		for j in range(self.paramDim):
			self.dWf += np.dot(dJ_dsl[j], x.T)
			self.dWf += np.dot(dJ_dsr[j], x.T)
		# L
		temp = np.dot(self.Wo.T, dJ_dso).flatten() + np.dot(self.Wi.T, dJ_dsi).flatten() + np.dot(self.Wu.T, dJ_dsu).flatten()
		for j in range(self.paramDim):
			temp += np.dot(self.Wf.T, dJ_dsl[j]).flatten()
			temp += np.dot(self.Wf.T, dJ_dsr[j]).flatten()
		self.dL[node.word] = temp 
		
		# Recursion
		for j in node.left:
			self.backProp(j, error_out)
		for j in node.right:
			self.backProp(j, error_out)
	else:
		x = np.reshape(self.L[:, node.word], (self.wvecDim, 1))
		dJ_dso = np.dot(dh_dso, error_at_h)
		dJ_dsi = np.dot(dc_dsi, error_at_c)
		dJ_dsu = np.dot(dc_dsu, error_at_c)
		# Bias
		self.dbo += dJ_dso.flatten()
		self.dbi += dJ_dsi.flatten()
		self.dbu += dJ_dsu.flatten()
		# Ws
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

if __name__ == '__main__':

    import trees as treeM
    f = open('data/trees')
    dataSize = 4
    mbData = []
    for j in range(dataSize):
    	mbData.append(treeM.Tree(f.readline().split('\t')[1].strip('\n')))
    
    #import tree as treeM
    #train = treeM.loadTrees()
    #numW = len(treeM.loadWordMap())

    wvecDim = 10
    middleDim = 10
    paramDim = 3
    numW = np.load('data/Lmat.npy').shape[0]

    rnn = TLSTM(wvecDim,middleDim,paramDim,numW,mbSize=4)
    rnn.initParams()

    #mbData = train[:4]
    #
    print "Numerical gradient check..."
    rnn.check_grad(mbData)

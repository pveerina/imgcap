import numpy as np
from collections import Counter

class Twin:
	def __init__(self, sentenceDim, imageDim, sharedDim, numLayers, scale, reg=1e-4, params=None):
		self.sentenceDim = sentenceDim
		self.imageDim = imageDim
		self.reg = reg
		self.sharedDim = sharedDim
		self.numLayers = numLayers
		self.scale = scale
		self.initParams(params)

	def initParams(self, params):

		self.sent_params = []
		self.sent_grads = []
		self.sent_biases = []
		self.sent_biasGrads = []
		self.img_params = []
		self.img_grads = []
		self.img_biases = []
		self.img_biasGrads = []

		def yb(): # yield bias matrix
			return np.zeros((self.sharedDim))
		def ybg(): # yield bias gradients
			return yb()
		def yW(): # yield W matrix
			return 0.1*np.random.randn(self.sharedDim, self.sharedDim)
		def yWg(): # yield W matrix gradients
			return np.zeros((self.sharedDim, self.sharedDim))


		self.sent_param_names = []
		self.sent_bias_names = []
		self.img_param_names = []
		self.img_bias_names = []

		# initialize sentence stuff
		self.sent_params.append(0.1*np.random.randn(self.sharedDim, self.sentenceDim))
		self.sent_grads.append(np.zeros((self.sharedDim, self.sentenceDim)))
		self.sent_biases.append(yb())
		self.sent_biasGrads.append(ybg())

		# initialize image stuff
		self.img_params.append(0.1*np.random.randn(self.sharedDim, self.imageDim))
		self.img_grads.append(np.zeros((self.sharedDim, self.imageDim)))
		self.img_biases.append(yb())
		self.img_biasGrads.append(ybg())
		self.sent_param_names.append('sent_W%i'%(0))
		self.sent_bias_names.append('sent_b%i'%(0))
		self.img_param_names.append('img_W%i'%(0))
		self.img_bias_names.append('img_b%i'%(0))
		# and for the remaining layers
		for l in xrange(self.numLayers):
			self.sent_params.append(yW())
			self.sent_param_names.append('sent_W%i'%(l+1))
			self.sent_grads.append(yWg())
			self.sent_biases.append(yb())
			self.sent_bias_names.append('sent_b%i'%(l+1))
			self.sent_biasGrads.append(ybg())
			self.img_params.append(yW())
			self.img_param_names.append('img_W%i'%(l+1))
			self.img_grads.append(yWg())
			self.img_biases.append(yb())
			self.img_bias_names.append('img_b%i'%(l+1))
			self.img_biasGrads.append(ybg())

		self.grads = self.sent_grads+self.sent_biasGrads+self.img_grads+self.img_biasGrads
		self.stack = self.sent_params+self.sent_biases+self.img_params+self.img_biases
		self.names = self.sent_param_names + self.sent_bias_names + self.img_param_names + self.img_bias_names

		if params is not None:
			for i, name in enumerate(self.names):
				self.stack[i] = params[name]

	def clearGradients(self):
		for y in self.grads:
			y[:] = 0.

	def costFunction(self, xs, ys, test=False):
		s1 = []
		s2 = []
		cost = 0.
		for i, (cy, cx) in enumerate(zip(ys, xs)):
			c1s = np.zeros(len(ys))
			c2s = np.zeros(len(ys))
			cpair = cx.dot(cy)
			for j, (iy, ix) in enumerate(zip(ys, xs)):
				if i != j:
					cpair = cx.dot(cy)
					c1s[j] = max(0, 1 - cpair + cx.dot(iy))
					c2s[j] = max(0, 1 - cpair + ix.dot(cy))
			s1.append(c1s)
			s2.append(c2s)
			cost += np.sum(c1s) + np.sum(c2s)
		if test:
			return cost
		xd = [] # x-deltas
		yd = [] # y-deltas
		for i, (cy, cx) in enumerate(zip(ys, xs)):
			c_xd = 0
			c_yd = 0
			for j, (iy, ix) in enumerate(zip(ys, xs)):
				if i != j:
					# delta w.r.t. x
					c_xd += (iy - cy)*(s1[i][j]>0) - cy*(s2[i][j]>0) + iy * (s2[j][i]>0)
					# delta w.r.t y
					c_yd += (ix - cx)*(s2[i][j]>0) - cx*(s1[i][j]>0) + ix * (s1[j][i]>0)
			xd.append(c_xd)
			yd.append(c_yd)
		return cost, xd, yd

	def newCostFunction(self, xs, ys, test=False):
	    xs = np.array(xs)
	    ys = np.array(ys)
	    s1 = xs.dot(ys.T).T
	    s2 = ys.dot(xs.T).T

	    s1 = np.maximum(0, 1 - np.diag(s1) + s1).T
	    s2 = np.maximum(0, 1 - np.diag(s2) + s2).T

	    s1[np.diag_indices_from(s1)] = 0
	    s2[np.diag_indices_from(s2)] = 0
	    ns1 = s1
	    ns2 = s2
	    cost = np.sum(s1)+np.sum(s2)
	    if test:
	    	return cost
	    s1t = s1 > 0
	    s2t = s2 > 0
	    tx1 = (ys[:,:,None].T - ys[:,:,None]).transpose([0,2,1])*s1t[:,:,None]
	    ty1 = (xs[:,:,None].T - xs[:,:,None]).transpose([0,2,1])*s2t[:,:,None]
	    tx2 = (ys * np.ones((len(xs),len(xs),xs[0].size))).transpose(1,0,2) * s2t[:,:,None]
	    ty2 = (xs * np.ones((len(xs),len(xs),xs[0].size))).transpose(1,0,2) * s1t[:,:,None]
	    tx3 = (s2t.T)[:,:,None]*ys
	    ty3 = (s1t.T)[:,:,None]*xs
	    xd = np.sum(tx1 - tx2 + tx3, 1)
	    yd = np.sum(ty1 - ty2 + ty3, 1)
	    return cost, list(xd), list(yd)

	def testCost(self, xs, ys):
		# tests the cost function deltas, to make sure they are
		# correct.
		#c, xd, yd = self.costFunction(xs, ys)
		c, xd, yd = self.newCostFunction(xs, ys)
		cyd = []
		cxd = []
		epsilon = 1e-5
		tol = 1e-5
		def rel_error(x, y):
		  """ returns relative error """
		  return np.max(np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y))))
		for i in xs:
		    ccxs = []
		    for j in range(i.size):
		        i[j] += epsilon / 2
		        #cP = self.costFunction(xs, ys, True)
		        cP = self.newCostFunction(xs, ys, True)
		        i[j] -= epsilon
		        #cN = self.costFunction(xs, ys, True)
		        cN = self.newCostFunction(xs, ys, True)
		        i[j] += epsilon / 2
		        ccxs.append((cP-cN)/epsilon)
		    cxd.append(np.array(ccxs))
		cxd = np.array(cxd)

		for i in ys:
		    ccxs = []
		    for j in range(i.size):
		        i[j] += epsilon / 2
		        #cP = self.costFunction(xs, ys, True)
		        cP = self.newCostFunction(xs, ys, True)
		        i[j] -= epsilon
		        #cN = self.costFunction(xs, ys, True)
		        cN = self.newCostFunction(xs, ys, True)
		        i[j] += epsilon / 2
		        ccxs.append((cP-cN)/epsilon)
		    cyd.append(np.array(ccxs))
		cyd = np.array(cyd)

		re1 = rel_error(xd, cxd)
		re2 = rel_error(yd, cyd)
		print 'Rel Error [x]: %g'%re1
		# if re1 > tol:
		# 	import pdb
		# 	pdb.set_trace()
		print 'Rel Error [y]: %g'%re2
		# if re2 > tol:
		# 	import pdb
		# 	pdb.set_trace()
		if re1 < tol and re2 < tol:
			return True
		return False

	def costAndGrad(self, mbdata, test=False, testCost=False):
		mbSize = len(mbdata)
		cost = 0.0
		batch_image_activations = []
		batch_sentence_activations = []
		self.clearGradients()
		for i, (imageVec, sentenceVec) in enumerate(mbdata):
			image_activations = self.forwardPropImage(imageVec)
			sentence_activations = self.forwardPropSentence(sentenceVec)
			batch_image_activations.append(image_activations)
			batch_sentence_activations.append(sentence_activations)
		sentActs = [x[-1] for x in batch_sentence_activations]
		imageActs = [x[-1] for x in batch_image_activations]
		if testCost:
			costCheckPass = self.testCost(imageActs, sentActs)
			if not costCheckPass:
				print 'Cost function grad check failed!'
			else:
				print 'Cost function grad check passed!'
		cost, image_deltas, sentence_deltas = \
		self.newCostFunction(imageActs, sentActs)
		if test:
			return cost

		img_input_grads = []
		sentence_input_grads = []
		voi = zip(range(len(image_deltas)), batch_image_activations, batch_sentence_activations)
		for i, image_activations, sentence_activations in voi:
			img_input_grad, sentence_input_grad = self.backwardProp( image_deltas[i], sentence_deltas[i], image_activations, sentence_activations)
			img_input_grads.append(img_input_grad)
			sentence_input_grads.append(sentence_input_grad)

		# remember to add in the L2-regularization for the parameters!
		for n in range(len(self.img_grads)):
			self.img_grads[n] += (self.reg / 2) * self.img_params[n]
			self.sent_grads[n] += (self.reg / 2) * self.sent_params[n]
			self.img_grads[n] *= self.scale
			self.img_biasGrads[n] *= self.scale
			self.sent_grads[n] *= self.scale
			self.sent_biasGrads[n] *= self.scale

		return cost, sentence_input_grads

	def forwardPropImage(self, imageVec):
		return self.forwardProp(imageVec.squeeze(), self.img_params, self.img_biases)

	def forwardPropSentence(self, sentVec):
		return self.forwardProp(sentVec.squeeze(), self.sent_params, self.sent_biases)

	def forwardProp(self, h, Ws, bs):
		activations = [h]
		for i in range(len(Ws)):
			h = np.maximum(Ws[i].dot(h).squeeze() + bs[i].squeeze(), 0)
			activations.append(h)
		return activations

	def backwardProp(self, image_delta_top, sentence_delta_top, image_activations, sentence_activations):

		# move backwards through the layers
		num_layers = len(self.sent_params)
		idx = range(num_layers)[::-1]
		# perform it for images first
		delta = image_delta_top
		h = image_activations
		for layer in idx:
			delta *= h[layer+1] > 0
			self.img_grads[layer] += np.outer(delta, h[layer])
			self.img_biasGrads[layer] += delta
			delta = delta.dot(self.img_params[layer])
			if layer == 0:
				# then return the error for the t-lstm
				image_input_grad = delta

		# now perform the same thing for sentences
		delta = sentence_delta_top
		h = sentence_activations
		for layer in idx:
			delta *= h[layer+1] > 0
			self.sent_grads[layer] += np.outer(delta, h[layer])
			self.sent_biasGrads[layer] += delta
			delta = delta.dot(self.sent_params[layer])
			if layer == 0:
				# then return the error for the t-lstm
				sent_input_grad = delta

		return image_input_grad, sent_input_grad


	def updateParams(self, scale, update):
		for i in xrange(len(self.stack)):
			self.stack[i] += scale * update[i]


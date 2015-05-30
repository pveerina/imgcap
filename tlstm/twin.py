import numpy as np
from collections import Counter

class Twin:
	def __init__(self, sentenceDim, imageDim, sharedDim, numLayers, scale, reg=1e-4):
		self.sentenceDim = sentenceDim
		self.imageDim = imageDim
		self.reg = reg
		self.sharedDim = sharedDim
		self.numLayers = numLayers
		self.scale = scale
		self.initParams()

	def initParams(self):

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

	def clearGradients(self):
		for y in self.grads:
			y[:] = 0.

	def costAndGrad(self, mbdata, test=False):
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

		image_deltas = []
		sentence_deltas = []
		cost = 0.0

		# compute cost
		ys = [x[-1] for x in batch_image_activations]
		xs = [x[-1] for x in batch_sentence_activations]
		s1 = []
		s2 = []
		# cy, cx = correct y, x
		# iy, ix = incorrect y, x
		for i, (cy, cx) in enumerate(zip(ys, xs)):
			c1s = []
			c2s = []
			cpair = cx.dot(cy)
			for j, (iy, ix) in enumerate(zip(ys, xs)):
				if i != j:
					cpair = cx.dot(cy)
					c1s.append(max(0, 1 - cpair + cx.dot(iy)))
					c2s.append(max(0, 1 - cpair + ix.dot(cy)))
			s1.append(c1s)
			s2.append(c2s)
			cost += np.sum(c1s) + np.sum(c2s)
		# now compute the deltas
		image_deltas = []
		sentence_deltas = []
		for i, (cy, cx) in enumerate(zip(ys, xs)):
			c_id = 0
			c_sd = 0
			for j, (iy, ix) in enumerate(zip(ys, xs)):
				if i != j:
					c_sd += (iy - cy)*(s1[i]>0) - cy*(s2[i]>0) + iy * (s2[j]>0)
					c_id += (ix - cx)*(s2[i]>0) - cx*(s1[i]>0) + ix * (s1[j]>0)
			image_deltas.append(c_id)
			sentence_deltas.append(c_sd)

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
		return self.forwardProp(imageVec, self.img_params, self.img_biases)

	def forwardPropSentence(self, sentVec):
		return self.forwardProp(sentVec, self.sent_params, self.sent_biases)

	def forwardProp(self, h, Ws, bs):
		activations = [h]
		for i, W in enumerate(Ws):
			h = np.maximum(W.dot(h).squeeze() + bs[i], 0)
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


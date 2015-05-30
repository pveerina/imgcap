import numpy as np
from collections import Counter

class Twin:
	def __init__(self, sentenceDim, imageDim, sharedDim, numLayers, reg=1e-4):
		self.sentenceDim = sentenceDim
		self.imageDim = imageDim
		self.reg = reg
		self.sharedDim = sharedDim
		self.numLayers = numLayers
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

		# and for the remaining layers
		for l in xrange(self.numLayers):
			self.sent_params.append(yW())
			self.sent_grads.append(yWg())
			self.sent_biases.append(yb())
			self.sent_biasGrads.append(ybg())
			self.img_params.append(yW())
			self.img_grads.append(yWg())
			self.img_biases.append(yb())
			self.img_biasGrads.append(ybg())

		self.gradStack = [self.sent_grads, self.sent_biasGrads, self.img_grads, self.img_biasGrads]
		self.stack = [self.sent_params, self.sent_biases, self.img_params, self.img_biases]

	def clearGradients(self):
		for y in self.gradStack:
			for x in y:
				x[:] = 0.

	def costAndGrad(self, mbdata, test=False):
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
		for i, (image_activations, sentence_activations) in enumerate(zip(batch_image_activations, batch_sentence_activations)):
			img_act = image_activations[-1]
			sent_act = sentence_activations[-1]
			correct_pair_cost = img_act.dot(sent_act)

			image_deltas.append(np.zeros_like(img_act))
			sentence_deltas.append(np.zeros_like(sent_act))

			for j in range(len(batch_image_activations)):
				if j != i:
					contrast_image_act = batch_image_activations[j][-1]
					contrast_sent_act = batch_sentence_activations[j][-1]
					s1 = max(0, 1 - correct_pair_cost + img_act.dot(contrast_sent_act))
					s2 = max(0, 1 - correct_pair_cost + contrast_image_act.dot(sent_act))
					image_deltas[i] -=  (sent_act * (s1 > 0))  + (sent_act + contrast_sent_act) * (s2 >0)
					sentence_deltas[i] -= (img_act * (s1 > 0)) + (img_act + contrast_image_act) * (s2>0)
					cost += s1 + s2
			# add in L2-regularization
			cost += self.reg * .5 * (np.sum([np.sum(x**2) for x in self.sent_params]) + np.sum([np.sum(x**2) for x in self.img_params]))

		img_input_grads = []
		sentence_input_grads = []
		voi = zip(range(len(image_deltas)), batch_image_activations, batch_sentence_activations)
		for i, image_activations, sentence_activations in voi:
			img_input_grad, sentence_input_grad = self.backwardProp( image_deltas[i], sentence_deltas[i], image_activations, sentence_activations)
			img_input_grads.append(img_input_grad)
			sentence_input_grads.append(sentence_input_grad)

		# remember to add in the L2-regularization for the parameters!
		for n in range(len(self.img_grads)):
			self.img_grads[n] += self.reg * self.img_params[n]
			self.sent_grads[n] += self.reg * self.sent_params[n]

		self.grads = {"img_grads": self.img_grads, \
				 "sent_grads": self.sent_grads, \
				 "img_biasGrads": self.img_biasGrads, \
				 "sent_biasGrads": self.sent_biasGrads}
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
		for i in xrange(len(self.params)):
			self.params[i] += scale * update["grads"][i]
			self.biases[i] += scale * update["biasGrads"][i]

		self.imageLayer += scale * update["imageLayerGrad"]
		self.imageBias += scale * update["imageLayerBiasGrad"]

		self.sentenceLayer += scale * update["sentenceLayerGrad"]
		self.sentenceBias += scale * update["sentenceLayerBiasGrad"]


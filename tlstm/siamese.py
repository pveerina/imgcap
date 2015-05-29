import numpy as np
from collections import Counter

class Siamese:

	def __init__(self, sentenceDim, imageDim, sharedDim, numLayers, reg=1e-4):
		self.sentenceDim = sentenceDim
		self.imageDim = imageDim
		self.reg = reg
		self.sharedDim = sharedDim
		self.numLayers = numLayers

	def initParams(self):
		
		self.params = []
		self.grads = []
		self.biases = []
		self.biasGrads = []

		self.imageLayer = 0.1*np.random.randn(self.sharedDim, self.imageDim)
		self.imageBias = np.zeros((self.imageDim))
		self.imageGrad = np.zeros((self.sharedDim, self.imageDim))
		self.imageBiasGrad = np.zeros((self.imageDim))
		
		self.sentenceLayer = 0.1*np.random.randn(self.sharedDim, self.sentenceDim)
		self.sentenceBias = np.zeros((self.sentenceDim))
		self.sentenceGrad = np.zeros((self.sharedDim, self.sentenceDim))
		self.sentenceBiasGrad = np.zeros((self.sentenceDim))

		for l in xrange(numLayers):
			self.params.append(0.1*np.random.randn(self.sharedDim, self.sharedDim))
			self.biases.append(np.zeros((self.sharedDim)))
			self.grads.append((np.zeros(self.sharedDim, self.sharedDim)))
	
	def clearGradients(self):
		self.grads = []
		self.biasGrads = []
		self.imageGrad = np.zeros((self.sharedDim, self.imageDim))
		self.imageBiasGrad = np.zeros((self.imageDim))
		self.sentenceGrad = np.zeros((self.sharedDim, self.sentenceDim))
		self.sentenceBiasGrad = np.zeros((self.sentenceDim))
		for l in xrange(numLayers):
			self.params.append(0.1*np.random.randn(self.sharedDim, self.sharedDim))
			self.biases.append(np.zeros((self.sharedDim)))
			self.grads.append((np.zeros(self.sharedDim, self.sharedDim)))
	
	def costAndGrad(self, mbdata, test=False):
		cost = 0.0
		batch_image_activations = []
		batch_sentence_activations = []
		labels = Counter()
		self.clearGradients()

		correct_pair_cost = 0.0
		for imageVec, sentenceVec, label in enumerate(mbdata):
			image_activations = self.forwardPropImage(imageVec)
			sentence_activations = self.forwardPropSentence(sentenceVec)
			batch_image_activations.append(image_activations)
			batch_sentence_activations.append(sentence_activations)

			correct_pair_cost += image_activation.dot(sentence_activation)

		# assumes that each image/caption pair is unique in the minibatch
		image_mismatch_cost = 0.0
		sent_mismatch_cost = 0.0
		for i in len(mbdata):
			for j in len(mbdata):
				if i != j:
					image_mismatch_cost += image_activations[i].dot(sentence_activations[j])
					sent_mismatch_cost += image_activations[j].dot(sentence_activations[i])

		cost = max(0, 1 - correct_pair_cost + image_mismatch_cost) + max(0, 1 - correct_pair_cost + sent_mismatch_cost)
		
		img_input_grads = []
		sentence_input_grads = []
		for image_activations, sentence_activations in zip(batch_image_activations, batch_sentence_activations):
			img_input_grad, sentence_input_grad = self.backwardProp(self, cost, image_activations, sentence_activations)
			img_input_grads.append(image_input_grad)
			sentence_input_grads.append(sentence_input_grad)

		grads = {"grads": self.grads, "biasGrads": self.biasGrads, "imageLayerGrad": self.imageGrad, "imageLayerBiasGrad":self.imageBiasGrad, "sentenceLayerGrad": self.sentenceGrad, "sentenceLayerBiasGrad":self.sentenceBiasGrad}
		return cost, img_input_grads, sentence_input_grads, grads


	def forwardPropImage(self, imageVec):
		i1 = np.maximum(np.dot(self.imageLayer, imageVec)) + self.imageBias, 0)

		image_activations = [i1]

		for i, W in enumerate(self.params):
			h = np.maximum(W.dot(image_activations[i-1]) + self.biases[i], 0)
			image_activations.append(h)

		return image_activations

	def forwardPropSentence(self, sentenceVec):
		s1 = np.maximum(np.dot(self.sentenceLayer, sentenceVec)) + self.sentenceBias, 0)
		sentence_activations = [s1]

		for i, W in enumerate(self.params):
			h = np.maximum(W.dot(sentence_activations[i-1]) + self.biases[i], 0)
			sentence_activations.append(h)

		return sentence_activations

	def backwardProp(self, cost, image_activations, sentence_activations):
		# backpropogate, storing gradients
		image_deltas = [0.0]
		sent_deltas = [0.0]

		num_layers = len(self.params)
		num_layers += 1

		image_input_grad = 0
		sentence_input_grad = 0
		# go backwards in layers and then the last l
		idx = reversed(range(num_layers))

		grads = []
		bias_grads = []

		for i in idx:
			image_delta = image_deltas[num_layers - i] * (image_activations[i] > 0)
			sent_delta = sent_deltas[num_layers - i] * (sentence_activations[i] > 0)

			if i == 1:
				self.imageGrad += image_delta.dot(image_activations[i-1])
				self.imageBiasGrad += image_delta
				image_input_grad = self.imageLayer.T.dot(image_delta)
				
				self.sentenceGrad += sent_delta.dot(sentence_activation[i-1])
				self.sentenceBiasGrad += sent_delta
				sentence_input_grad = self.sentenceLayer.T.dot(sent_delta)
				break
			else:
				dwi = image_delta.dot(image_activations[i-1])
				dbi = image_delta
				image_deltas.append(self.params[i].T.dot(image_delta))

				dws = sent_delta.dot(sentence_activations[i-1])
				dbs = sent_delta
				sent_deltas.append(self.params[i].T.dot(sent_delta))

				grads.append(dwi+dws)
				bias_grads.append(dbi+dbs)
		self.grads = reverse(grads)
		self.biasGrads = reverse(bias_grads)
		return image_input_grad, sentence_input_grad

	def updateParams(self, scale, update):

		for i in xrange(len(self.params)):
			self.params[i] += scale * update["grads"][i]
			self.biases[i] += scale * update["biasGrads"][i]

		self.imageLayer += scale * update["imageLayerGrad"]
		self.imageBias += scale * update["imageLayerBiasGrad"]

		self.sentenceLayer += scale * update["sentenceLayerGrad"]
		self.sentenceBias += scale * update["sentenceLayerBiasGrad"]
		
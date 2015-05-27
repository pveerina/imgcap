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

		self.imageLayer = 0.1*np.random.randn(self.sharedDim, self.imageDim)
		self.imageBias = np.zeros((self.imageDim))
		self.imageGrad = np.zeros((self.sharedDim, self.imageDim))
		self.sentenceLayer = 0.1*np.random.randn(self.sharedDim, self.sentenceDim)
		self.sentenceBias = np.zeros((self.sentenceDim))
		self.sentenceGrad = np.zeros((self.sharedDim, self.sentenceDim))

		for l in xrange(numLayers):
			self.params.append(0.1*np.random.randn(self.sharedDim, self.sharedDim))
			self.biases.append(np.zeros((self.sharedDim)))
			self.grads.append((np.zeros(self.sharedDim, self.sharedDim)))


	def costAndGrad(self, mbdata, test=False):

		cost = 0.0
		image_activations = []
		sentence_activations = []
		labels = Counter()

		correct_pair_cost = 0.0
		for imageVec, sentenceVec, label in enumerate(mbdata):
			image_activation = self.forwardPropImage(imageVec)
			sentence_activation = self.forwardPropSentence(sentenceVec)
			image_activations.append(image_activation)
			sentence_activations.append(sentence_activation)

			correct_pair_cost += image_activation.dot(sentence_activation)

		# assumes that each image/caption pair is unique in the minibatch
		image_mismatch_cost = 0.0
		sent_mismatch_cost = 0.0
		for i in len(mbdata):
			for j in len(mbdata):
				if i != j:
					image_mismatch_cost += image_activations[i].dot(sentence_activations[j])
					sent_mismatch_cost += image_activations[j].dot(sentence_activations[i])

		cost = max(0, 1 - correct_pair_cost + image_mismatch_cost) + 
		max(0, 1 - correct_pair_cost + sent_mismatch_cost)

		self.backwardProp(self, cost, image_activations, sentence_activations)
		



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
		pass

	def updateParams(self, scale, update):
		#update params
		pass


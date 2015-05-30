# Computes gradient check for `twin.py'
# IMPORTANT: Expects `twin.py' to be in the same folder.
import numpy as np
from twin import Twin
def check_grad(net, data, epsilon=1e-6):
	cost, grad = net.costAndGrad(data)
	err1 = 0.0
	count1  = 0
	print "Checking gradients..."
	for list_W, list_dW in zip(net.stack, net.grads):
		for i in range(len(list_W)):
			W = list_W[i]
			dW = list_dW[i]
			W = W[..., None]
			dW = dW[..., None]
			err2 = 0.0
			count2 = 0
			flag = None
			for j in xrange(W.shape[0]):
				for k in xrange(W.shape[1]):
					W[j, k] += epsilon
					costP,_ = net.costAndGrad(data)
					W[j, k] -= epsilon
					numGrad = (costP-cost)/epsilon
					err = np.abs(dW[j, k] - numGrad)
					err1 += err
					err2 += err
					count1 += 1
					count2 += 1
					if np.sign(dW[j, k]) != np.sign(numGrad):
						flag = "X"
			print err2/count2, np.sign(dW[j, k]), np.sign(numGrad)


def generate_fake_data(numVectors, vecDim):
	mbdata = []
	for i in xrange(numVectors):
		a = np.random.randn(vecDim, 1)
		b = np.random.randn(vecDim, 1)
		mbdata.append([a, b])
	return mbdata

numVectors = 10
vecDim = 100
fake_data = generate_fake_data(numVectors, vecDim)
sentenceDim = vecDim
imageDim = vecDim
sharedDim = 10
numLayers = 3
scale = 1
net = Twin(sentenceDim, imageDim, sharedDim, numLayers, scale)
net.initParams()
check_grad(net, fake_data)

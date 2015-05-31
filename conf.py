'''
This file stores the configuration for the entire network, as this is
simply much easier than passing in all of the configurations as
arguments to be parsed by the wrapper.
'''
import os
# global parameters
#root = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap' # root directory
root = os.path.dirname(os.path.realpath(__file__))
alpha = 1e-2 # the learning rate
# 1e-2 is the fastest the vanilla network can learn using SGD and no
# regularization. it appears to be able to work with ADAGRAD as well,
# but the long-term behavior hasn't been checked yet.
#
# the network also appears to learn fine using an alpha of 1e-2 even
# when using the default regularization, which is a good sign.
optimizer = 'sgd'

# data handler parameters
megabatch_size = 10000 # the size of the chunk to load into memory at a time
minibatch_size = 10 # the batch size for SGD
val_size = 0.05 # the fraction of the data to use for validation
test_size = 0.05 # the fraction of the data to use for testing
data_type = 'vgg16' # the image features to load, {vgg16 | vgg19 | both}
epoch_lim = 50 # the number of epochs to run for

# DT-LSTM network parameters
wvecDim = 300 # dimensionality of the word vectors
middleDim = 250 # dimensionality of the hidden layer
paramDim = 2 # the number of children to consider is given by
             # (2*paramDim), any children beyond this are not considered
numWords = None # the size of the L matrix, defined by the runNet script
mbSize = minibatch_size # the minibatch size
rho = 1e-4 # the regularization constant

# twin network parameters
sentenceDim = middleDim # the dimensionality of the sentences
imageDim = None # the dimensionality of the images; set by runNet
sharedDim = 275 # the size of the final output
numLayers = 2 # the number of layers to include, not including the first layer
reg = 1e-4 # the regularization constant

import numpy as np
from tlstm.datahandler import DataHandler
import optparse
import conf as opts
from tlstm.tlstm import TLSTM
from tlstm.twin import Twin
from tlstm import sgd as optimizer
from matplotlib import pyplot as plt

def makeconf(conf_arr):
    # makes a confusion matrix plot when provided a matrix conf_arr
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width = len(conf_arr)
    height = len(conf_arr[0])

    cb = fig.colorbar(res)
    indexs = '0123456789'
    plt.xticks(range(width), indexs[:width])
    plt.yticks(range(height), indexs[:height])
    # you can save the figure here with:
    # plt.savefig("pathname/image.png")

    plt.show()

k = ['mscoco_441795', 'mscoco_292051', 'mscoco_109212', 'mscoco_205103', 'mscoco_2006', 'mscoco_460458', 'mscoco_370813', 'mscoco_136849', 'mscoco_221313', 'mscoco_67748']

# ensure the options are valid
assert opts.megabatch_size % opts.minibatch_size == 0
assert type(opts.data_type) == str
opts.data_type = opts.data_type.lower()
assert opts.data_type in ['vgg16','vgg19','both']

test_mode = True

# set opts that have only one possible value
opts.numWords = 33540
opts.imageDim = 4096
if opts.data_type == 'both':
    opts.imageDim *= 2

# instantiate the data handler
dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)

idx = []
for i in range(len(k)):
    idx.append("#"+("%02d"%(np.random.randint(5))))
for i in range(len(k)):
    k[i]+=idx[i]
b = dh.constructBatch(k)

if opts.saved_model is not None:
	params = np.load(opts.saved_model)
else:
	params = None

# instantiate the second 'layer'
net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), opts.reg, params=params)

# instantiate the first 'layer'
net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), opts.rho, net2, root=opts.root, params=params)

cost, total, xs, ys = net1.costAndGrad(b, test=True)
scores = np.dot(xs, ys.T)
makeconf(scores)

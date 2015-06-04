# get the top whatever at whatever
import os
import imp
import numpy as np
from tlstm.datahandler import DataHandler
from tlstm.twin import Twin
from tlstm.tlstm import TLSTM

#####
##### specify your folder here
#####
folder = '/afs/.ir.stanford.edu/users/n/d/ndufour/public/imgcap/models/m_0602_163204'
#####
##### specify the model to use here
#####
model = 'megabatch_8_epoch1.npz'

opts = imp.load_source('opts',os.path.join(folder, 'config'))
opts.root = '/'.join(opts.root.split('/')[:-2])

opts.numWords = 33540
opts.imageDim = 4096
if opts.data_type == 'both':
    opts.imageDim *= 2

val = eval(open(os.path.join(folder, 'val'),'r').read())

dh = DataHandler(opts.root, opts.megabatch_size, opts.minibatch_size, opts.val_size, opts.test_size, opts.data_type, opts.epoch_lim)

# we have to create val sets
def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

print 'Rehydrating val sets'
val_ims = []
val_capts = []
for i in val:
    for desc in dh.data_dict[i]['desc_ids']:
        val_ims.append(i)
        val_capts.append(desc)
b = dh.constructBatch(val_capts)

print 'Rehydrating neural nets'
params = np.load(os.path.join(folder, model))

net2 = Twin(opts.sentenceDim, opts.imageDim, opts.sharedDim, opts.numLayers, 1./(opts.mbSize*(opts.mbSize-1)), opts.reg, params=params)

net1 = TLSTM(opts.wvecDim, opts.middleDim, opts.paramDim, opts.numWords, opts.mbSize, 1./(opts.mbSize*(opts.mbSize-1)), opts.rho, net2, root=opts.root, params=params)

xsl = []
ysl = []

for n,x in enumerate(chunks(b, 10)):
    print n
    c, _, xs, ys = net1.costAndGrad(x, test=True)
    xsl+=list(xs) # sentences
    ysl+=list(ys) # images

# we also have to construct a mapping from image_idx --> desc_idx
# and desc_idx --> img_idx
xs = np.array(xsl)
ys = []
img2desc_map = dict()
desc2img_map = dict()
img_idx = -1
cap_idx = -1
vi2 = []
for n,(im, desc) in enumerate(zip(val_ims, val_capts)):
    if not img2desc_map.has_key(im):
        img_idx = len(ys)
        img2desc_map[im] = []
        vi2.append(im)
        ys.append(ysl[n])
    img2desc_map[im].append(n)
    desc2img_map[desc] = img_idx

ys = np.array(ys)

res = xs.dot(ys.T)

ca_res = np.zeros_like(res)

for i in xrange(res.shape[0]):
    print i
    q = np.argsort(res[i,:])[::-1] # images for this caption
    j = list(q).index(desc2img_map[val_capts[i]])
    ca_res[i,j] += 1

act_tot = 0
im_res = np.zeros_like(res)
for j in xrange(res.shape[1]):
    print j
    q = list(np.argsort(res[:,j])[::-1]) # captions for this image
    sq = set(q)
    idxs = []
    for zp in img2desc_map[vi2[j]]:
        idxs.append(q.index(zp))
    i = min(idxs)
    im_res[i,j] += 1

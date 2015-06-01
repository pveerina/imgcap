# this exports a class that yields 'megabatches', which are
# efficiently sampled from the various data sources that we have
#
# it requires the parameters:
#
# data_type
#   either 'VGG19' or 'VGG16' or 'both'
#
# megabatch_size
#   megabatches are the 'big batches' of data, which are then
#   then partitioned into smaller batches. These are expressed
#   in terms of *images*
#
# minibatch_size
#   the minibatches are partitions of each megabatch, which is fetched
#   automatically behind-the-scenes. these are expressed in terms of
#   *datums* (i.e., image / desc pairs)
#
# val_size
# test_size
#
# Note: val_size and test_size don't refer to the number of
# testing/validation *items* to include, but rather the number of images
# to place into each set. Since images have a variable number of
# descriptions each, this can potentially be wildly variable.
#
# when initialized, it elaborates the image-wise global mapping into
# a dictionary that stores all relevant data.

import os
import numpy as np
from trees import Tree

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class DataHandler():
    def __init__(self, root=None, megabatch_size=10000, minibatch_size=10, val_size=.05, test_size=.05, data_type='vgg16', epoch_lim=10):

        if root == None:
            # assume you're running from the lstm directory
            self.root = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/'
        else:
            self.root = root
        if type(data_type) != str:
            print 'Data type not understood'
            return
        if data_type.lower() != 'vgg19' and data_type.lower() != 'vgg16' and data_type.lower() != 'both':
            print 'Data type must be \'vgg16\' or \'vgg19\' or \'both\''
            return
        self.megabatch_size, self.minibatch_size, self.val_size, self.test_size, self.data_type, self.epoch_lim = megabatch_size, minibatch_size, val_size, test_size, data_type, epoch_lim
        self.dataroot = os.path.join(self.root, 'data')
        datafile = os.path.join(self.dataroot, 'total_ordering_imagewise')
        treefile = os.path.join(self.dataroot, 'trees','trees')

        # the data_dict stores all relevant information. This gets initialized
        # only once, and is indexed by image id
        data_dict = dict()

        # load in all the data files as memory-maps
        print 'Memory mapping image features'
        img_feats = dict()
        img_feats['vgg16'] = dict()
        img_feats['vgg16']['coco_train_VGG16_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'coco_train_VGG16_feats.npy'),mmap_mode='r')
        img_feats['vgg16']['coco_val_VGG16_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'coco_val_VGG16_feats.npy'),mmap_mode='r')
        img_feats['vgg16']['flickr30k_VGG16_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'flickr30k_VGG16_feats.npy'),mmap_mode='r')

        img_feats['vgg19'] = dict()
        img_feats['vgg19']['coco_train_VGG19_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'coco_train_VGG19_feats.npy'),mmap_mode='r')
        img_feats['vgg19']['coco_val_VGG19_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'coco_val_VGG19_feats.npy'),mmap_mode='r')
        img_feats['vgg19']['flickr30k_VGG19_feats.npy'] = np.load(os.path.join(self.dataroot, 'image_features', 'flickr30k_VGG19_feats.npy'),mmap_mode='r')
        self.img_feats = img_feats

        print 'Loading trees'
        self.trees = [x.split('\t')[1] for x in open(treefile, 'r').read().strip().split('\n')]

        print 'Constructing data attribute dictionary'

        with open(datafile, 'r') as f:
            for line in f:
                elem = line.strip().split('\t')
                cdata = dict()
                nomen = elem[0]
                cdata['vgg16'] = elem[1]
                cdata['vgg19'] = elem[2]
                cdata['img_feat_idx'] = int(elem[3])
                cdata['img_fn'] = elem[4]
                cdata['desc_idx'] = []
                cdata['desc_ids'] = []
                cdata['n_desc'] = 0
                elem = elem[5:]
                for n, i in enumerate(elem):
                    if not n%2:
                        cdata['desc_ids'].append(i)
                        cdata['n_desc'] += 1
                    else:
                        cdata['desc_idx'].append(int(i))
                data_dict[nomen] = cdata

        # compute the actual test and validation sizes
        if test_size > 0 and test_size < 1:
            self.test_size = int(test_size * len(data_dict))
        if val_size > 0 and val_size < 1:
            self.val_size = int(val_size * len(data_dict))
        self.test_size = int(self.test_size)
        self.val_size = int(self.val_size)
        # select the image set for testing
        shuf_ims = data_dict.keys()
        np.random.shuffle(shuf_ims)
        self.test_ims = shuf_ims[:self.test_size]
        self.val_ims = shuf_ims[self.test_size:(self.test_size + self.val_size)]
        self.train_ims = shuf_ims[(self.test_size + self.val_size):]
        self.data_dict = data_dict

        self.cur_epoch = 0
        self.cur_megabatch = 0
        self.cur_iteration = 0

        self.megabatch_queue = []
        self.minibatch_queue = []
        self.testing = False
        self.test_megabatch_queue = []
        self.test_minibatch_queue = []
        self.batchPerEpoch = None
    def saveSets(self, folder):
        with open(os.path.join(folder, 'train'),'w') as f:
            f.write(str(self.train_ims))
        with open(os.path.join(folder, 'val'),'w') as f:
            f.write(str(self.val_ims))
        with open(os.path.join(folder, 'test'),'w') as f:
            f.write(str(self.test_ims))
    def nextBatch(self, test=False):
        # yields the next batch
        if test and not self.testing:
            self.testing = True
            self.testEpoch()
        if self.testing:
            megabatch_queue = self.test_megabatch_queue
            minibatch_queue = self.test_minibatch_queue
        else:
            megabatch_queue = self.megabatch_queue
            minibatch_queue = self.minibatch_queue
        if not len(minibatch_queue) and not len(megabatch_queue):
            if self.testing:
                print 'Test Epoch complete'
                self.testing = False
                return -1
            if self.cur_epoch == self.epoch_lim:
                print 'Done'
                return None
            self.epochAdvance()
            self.cur_megabatch = 0
            self.megabatchAdvance()
            self.cur_iteration = 0
        elif not len(minibatch_queue):
            self.megabatchAdvance()
        if not self.testing:
            self.cur_iteration += 1
        return minibatch_queue.pop(0)
    def testEpoch(self):
        print 'Beginning test epoch'
        np.random.shuffle(self.val_ims)
        self.test_megabatch_queue = [x for x in chunks(self.val_ims, self.megabatch_size)]
        print '\t%i megabatches'%(len(self.test_megabatch_queue))
    def epochAdvance(self):
        self.cur_epoch += 1
        print 'Beginning epoch %i'%self.cur_epoch
        np.random.shuffle(self.train_ims)
        self.megabatch_queue = [x for x in chunks(self.train_ims, self.megabatch_size)]
        print '\t%i megabatches'%(len(self.megabatch_queue))
    def megabatchAdvance(self):
        if self.testing:
            self.testMegabatch()
            return
        print 'Loading megabatch data'
        self.cur_megabatch += 1
        img_dat = dict()
        tree_dat = dict()
        tree_rem = dict()
        cur_queue = self.megabatch_queue.pop(0)
        for i in cur_queue:
            tree_rem[i] = self.data_dict[i]['n_desc']
            vgg16 = self.img_feats['vgg16'][self.data_dict[i]['vgg16']]
            vgg19 = self.img_feats['vgg19'][self.data_dict[i]['vgg19']]
            imidx = self.data_dict[i]['img_feat_idx']
            if self.data_type == 'both':
                img_dat[i] = np.hstack((vgg16[imidx,:], vgg19[imidx,:]))
            elif self.data_type == 'vgg16':
                img_dat[i] = vgg16[imidx,:]
            elif self.data_type == 'vgg19':
                img_dat[i] = vgg19[imidx,:]
            tree_dat[i] = []
            for curtreeIDX in self.data_dict[i]['desc_idx']:
                tree_dat[i].append(Tree(self.trees[curtreeIDX]))
            # shuffle the trees
            np.random.shuffle(tree_dat[i])
        print 'Constructing schedule for this megabatch'
        while len(tree_rem) >= self.minibatch_size:
            sample = np.random.choice(tree_rem.keys(), self.minibatch_size, replace=False)
            for k in sample:
                tree_rem[k]-=1
                if tree_rem[k] == 0:
                    tree_rem.pop(k, None)
            sample = [[img_dat[x], tree_dat[x].pop(0)] for x in sample]
            self.minibatch_queue.append(sample)
        if self.batchPerEpoch == None:
            self.batchPerEpoch = len(self.minibatch_queue) * (len(self.megabatch_queue) + 1)
        print 'Beginning megabatch %i (epoch %i)'%(self.cur_megabatch, self.cur_epoch)
    def testMegabatch(self):
        print 'Loading test megabatch data'
        img_dat = dict()
        tree_dat = dict()
        tree_rem = dict()
        cur_queue = self.test_megabatch_queue.pop(0)
        for i in cur_queue:
            tree_rem[i] = self.data_dict[i]['n_desc']
            vgg16 = self.img_feats['vgg16'][self.data_dict[i]['vgg16']]
            vgg19 = self.img_feats['vgg19'][self.data_dict[i]['vgg19']]
            imidx = self.data_dict[i]['img_feat_idx']
            if self.data_type == 'both':
                img_dat[i] = np.hstack((vgg16[imidx,:], vgg19[imidx,:]))
            elif self.data_type == 'vgg16':
                img_dat[i] = vgg16[imidx,:]
            elif self.data_type == 'vgg19':
                img_dat[i] = vgg19[imidx,:]
            tree_dat[i] = []
            for curtreeIDX in self.data_dict[i]['desc_idx']:
                tree_dat[i].append(Tree(self.trees[curtreeIDX]))
            # shuffle the trees
            np.random.shuffle(tree_dat[i])
        print 'Constructing schedule for this testing megabatch'
        while len(tree_rem) >= self.minibatch_size:
            sample = np.random.choice(tree_rem.keys(), self.minibatch_size, replace=False)
            for k in sample:
                tree_rem[k]-=1
                if tree_rem[k] == 0:
                    tree_rem.pop(k, None)
            sample = [[img_dat[x], tree_dat[x].pop(0)] for x in sample]
            self.test_minibatch_queue.append(sample)
        print 'Beginning testing megabatch'



# we need a unified file description method:
#
# ALL FIELDS SEPARATED BY TABS
#
# caption_id
# caption_idx
# image_feat_file (vgg16)
# image_feat_file (vgg19)
# image_feat_idx
# image_filename
#
import os

droot = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/'
trees = os.path.join(droot, 'trees', 'trees')

img_idx_files = dict()
img_idx_files['coco'] = ['image2idx/coco_train_imgs.txt', 'image2idx/coco_val_imgs.txt']
img_idx_files['flickr'] = ['image2idx/flickr30k-images-list.txt']


def get_desc_type(descidx):
    return descidx.split('_')[0]

def get_img_file(descidx):
    dtype = get_desc_type(descidx)
    if dtype=='mscoco':
        return img_idx_lists['coco']
    elif dtype == 'flickr':
        return img_idx_lists['flickr']

def get_img_id(descidx):
    return descidx.split('_')[1].split('#')[0]

def get_img_fname(descidx):
    dtype = get_desc_type(descidx)
    dnum = get_img_id(descidx)
    if dtype == 'flickr':
        return ['%s'%dnum]
    elif dtype == 'mscoco':
        template = 'COCO_%s2014_%012i.jpg'
        return [template%('train', int(dnum)), template%('val',int(dnum))]

def get_img_idx(descidx):
    # returns the filename, type type, the index, and the index of the file
    dtype = get_desc_type(descidx)
    fnames = get_img_fname(descidx)
    lists = get_img_file(descidx)
    for n, l in enumerate(lists):
        for fn in fnames:
            if fn in l:
                return (dtype, fn, l.index(fn), n)



# read in desc ids and get ns
descs = dict()
descline = open(trees, 'r').read().strip().split('\n')
for n,line in enumerate(descline):
    cid = line.split('\t')[0]
    descs[cid] = n

img_idx_lists = dict()
for k in img_idx_files.keys():
    elist = []
    for f in img_idx_files[k]:
        tmp = open(os.path.join(droot, f),'r').read().strip().split('\n')
        elist.append(tmp)
    img_idx_lists[k] = elist

# caption_id
# caption_idx
# image_feat_file (vgg16)
# image_feat_file (vgg19)
# image_feat_idx
# image_filename

total_ordering = []
for asdf, k in enumerate(descs.keys()):
    print '%i/%i'%(asdf, len(descs))
    tmptup = []
    tmptup.append(k)
    tmptup.append(descs[k])
    dtype, fn, imgidx, lidx = get_img_idx(k)
    if dtype == 'mscoco':
        # then look for it
        if 'train' in img_idx_files['coco'][lidx]:
            template = 'coco_train_%s_feats.npy'
        elif 'val' in img_idx_files['coco'][lidx]:
            template = 'coco_val_%s_feats.npy'
        tmptup.append(template%'VGG16')
        tmptup.append(template%'VGG19')
    elif dtype == 'flickr':
        tmptup.append('flickr30k_%s_feats.npy'%'VGG16')
        tmptup.append('flickr30k_%s_feats.npy'%'VGG19')
    tmptup.append(imgidx)
    tmptup.append(fn)
    total_ordering.append(tmptup)

to_write = []
for a,b,c,d,e,f in total_ordering:
    to_write.append('%s\t%i\t%s\t%s\t%i\t%s'%(a,b,c,d,e,f))
with open(os.path.join(droot, 'total_ordering'),'w') as f:
    f.write('\n'.join(to_write))


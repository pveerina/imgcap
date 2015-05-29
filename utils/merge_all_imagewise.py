# this replicates the functionality of merge_all, but is indexed by
# image. The fields now are:
#
# 0 photo_id
# 1 image_feat_file (vgg16)
# 2 image_feat_file (vgg19)
# 3 image_feat_idx
# 4 image_filename
# 5 caption_id_0
# 6 caption_idx_0
# ...
# x caption_id_n
# y caption_idx_n
#
# (old system):
# 0 caption_id
# 1 caption_idx
# 2 image_feat_file (vgg16)
# 3 image_feat_file (vgg19)
# 4 image_feat_idx
# 5 image_filename

ofn = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/total_ordering'
nfn = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/total_ordering_imagewise'
nfd = dict()

with open(ofn, 'r') as f:
    for line in f:
        elem = line.strip().split('\t')
        imid = elem[0].split('#')[0]
        if not nfd.has_key(imid):
            nfd[imid] = [imid, elem[2], elem[3], elem[4], elem[5]]
        if not nfd[imid][1] == elem[2]:
            print 'Error!'
            break
        if not nfd[imid][4] == elem[5]:
            print 'Error!'
            break
        nfd[imid].append(elem[0])
        nfd[imid].append(elem[1])

to_write = []
for k in nfd.keys():
    klist = [str(x) for x in nfd[k]]
    to_write.append('\t'.join(klist))

with open(nfn, 'w') as f:
    f.write('\n'.join(to_write))



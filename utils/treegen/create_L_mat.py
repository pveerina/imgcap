# this creates a matrix of L-features and also re-writes the dep_parse
# as a series of tuples (x, xi, y, yi) where x is the index of the
# governor, y is the index of the dependent, and xi, yi are their
# respective indices in the original sentence--note: ROOT is represented
# by -1
#
# the indices correspond to the values of the matrix L_mat, which this
# also creates.

from collections import defaultdict as ddict
import cPickle

parse_data = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen/dep_parse_prep'

Ldictf = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/L_dict'

Ldict = cPickle.load(open(Ldictf,'r'))

vec_set = []
word_map = dict()

for n,k in enumerate(Ldict.keys()):
    word_map[k] = n
    vec_set.append(Ldict[k])

f = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/word2index'
with open(f,'w') as f:
    to_write = []
    for k in word_map.keys():
        to_write.append('%s\t%i'%(k, word_map[k]))
    f.write('\n'.join(to_write))

wm = ddict(lambda: wm['UUUNKKK'], word_map)

to_write = []
r = open(parse_data, 'r').read().strip().split('\n')
for line in r:
    cid, tree = line.split('\t')
    tree = eval(tree)
    ntree = []
    c = tree[0]
    ntree.append((-1, 0, wm[c[1][0]], c[1][1]))
    for c in tree[1:]:
        ntree.append((wm[c[0][0]], c[0][1], wm[c[1][0]], c[1][1]))
    to_write.append('%s\t%s'%(cid, str(ntree)))

f = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/trees'
with open(f, 'w') as f:
    f.write('\n'.join(to_write))

vec_set = np.array(vec_set)
np.save('/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/Lmat',vec_set)

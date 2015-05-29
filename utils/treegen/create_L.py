# we need to create an 'L' file from GLoVe such that we don't have to
# search through all of the vectors for words.
from collections import Counter

glove = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/glove.840B.300d.txt'

parse_data = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen/dep_parse_prep'


# let's compute the arity distribution of the trees
trees = [x.split('\t')[1] for x in open(parse_data,'r').read().strip().split('\n')]


all_words = set()

for n,tree in enumerate(trees):
    print n
    tree = eval(tree)
    # each tree consists of a series of tuples
    words = set(reduce(lambda x, y: x+y, [[x[0][0], x[1][0]] for x in tree[1:]]))
    all_words.update(words)

glove_vecs = dict()

with open(glove,'r') as f:
    for n,line in enumerate(f):
        print n
        elems = line.split(' ')
        if elems[0] in all_words:
            glove_vecs[elems[0]] = [float(x) for x in elems[1:]]

# 'UUUNKKK' should be a random vector sampled from the distribution of GLoVes
#
# glove mean ~ 0
# glove std = 0.3867
#
# and is otherwise a perfectly normal distribution
k = np.array(glove_vecs.values()).reshape(-1)
glove_vecs['UUUNKKK'] = (np.random.randn(300)*np.std(k)) + np.mean(k)

from collections import defaultdict as ddict()

ngv = ddict(lambda: ngv['UUUNKKK'])

L = ddict(lambda: L['UUUNKKK'], glove_vecs)
import cPickle

with open('L_mat','w') as f:
    cPickle.dump(glove_vecs, f)

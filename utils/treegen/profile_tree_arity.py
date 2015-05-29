# we need to create an 'L' file from GLoVe such that we don't have to
# search through all of the vectors for words.
from collections import Counter

glove = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/glove.840B.300d.txt'

parse_data = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen/dep_parse_prep'


# let's compute the arity distribution of the trees
trees = [x.split('\t')[1] for x in open(parse_data,'r').read().strip().split('\n')]


nc = Counter() # number of children
nlc = Counter() # number of left children
nrc = Counter() # number of right children
for n,tree in enumerate(trees):
    print n
    tree = eval(tree)
    # each tree consists of a series of tuples
    words = set(reduce(lambda x, y: x+y, [[x[0][0], x[1][0]] for x in tree[1:]]))
    cdict = dict()
    for w in words:
        cdict[w] = [0, 0] # n left, right children
    for i in tree[1:]:
        w = i[0][0]
        w1idx = i[0][1]
        w2idx = i[1][1]
        if w1idx > w2idx:
            cdict[w][0] += 1
        else:
            cdict[w][1] += 1
    for i in cdict.values():
        nc[i[0]+i[1]] += 1
        nlc[i[0]] += 1
        nrc[i[1]] += 1

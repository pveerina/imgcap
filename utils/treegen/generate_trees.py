# Accepts a document of the form:
#
# <img_id> (tab) <img_description> (return)
#
# and then generates a tree of each img_description, tokenizing the
# words as it goes. it maintains a persistent word tokenization
# dictionary, called mapping, so that you can call it on an arbitrary
# number of files and there will be no word collisions. The parse trees
# are also maintained in a continuous file.
import os
import cPickle
from nltk.parse import stanford
from nltk.tree import Tree
from multiprocessing import Process, Queue

# You will need to customize this
PARSER_LOC = '/Users/ndufour/stanford-parser/stanford-parser.jar'
MODEL_LOC = '/Users/ndufour/stanford-parser/stanford-parser-3.5.2-models.jar'
PCFG_LOC = '/Users/ndufour/stanford-parser/stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'

root = os.path.dirname(os.path.abspath(__file__))

def get_map():
    # returns the current mapping
    if not os.path.exists(os.path.join(root, 'mapping')):
        return dict()
    else:
        f = open(os.path.join(root, 'mapping'),'r').read()
        mapping = dict()
        for i in f.split('\n'):
            if len(i):
                spl = i.split('\t')
                mapping[spl[0]] = int(spl[1])

def seen_ids():
    # returns a set of IDs tha thave already been seen by the parser
    if not os.path.exists(os.path.join(root, 'trees.txt')):
        return set()
    else:
        sid = set()
        f = open(os.path.join(root, 'trees.txt')).read().split('\n')
        for i in f:
            if not len(i):
                continue
            cid, _ = i.split('\t')
            sid.add(cid)
        return sid

mapping = get_map()
seen = seen_ids()

def init_parser():
    # initializes the parser
    os.environ['STANFORD_PARSER'] = PARSER_LOC
    os.environ['STANFORD_MODELS'] = MODEL_LOC
    parser = stanford.StanfordParser(model_path=PCFG_LOC)
    return parser

def gk(item):
    if not mapping.has_key(item):
        with open(os.path.join(root, 'mapping'),'a') as mapf:
            v = item
            k = len(mapping)
            mapping[v] = k
            mapf.write('%s\t%i\n'%(v,k))
    return mapping[item]

def worker(qIN, qOUT):
    # queue worker; instantiates its own stanford parse, then dequeues
    # sentences from qIN, parses them, adds them to qOUT.
    # format of each qIN item:
    # <id> <sentence> <num> <tot_num>
    parser = init_parser()
    while True:
        qr = qIN.get()
        if qr == None:
            return
        cid, sent, num, tot = qr
        print '%i / %i : %s'%(num, tot, sent)
        parse = parser.raw_parse(sent).next()
        qOUT.put([cid, str(parse)])

def get_str(tree):
    if type(tree)==unicode or type(tree)==str:
        return str(gk(tree))
    if len(tree) == 1:
        return get_str(tree[0])
    return '(' + ' '.join([get_str(x) for x in tree]) + ')'

parser = init_parser()
filename = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/flickr_30k_v20130124.token'
f = open(filename, 'r').read().split('\n')

# with open(os.path.join(root, 'trees.txt'),'a') as treefile:
#     for n,line in enumerate(f):
#         if not len(line):
#             continue
#         cid, descr = line.split('\t')
#         print '%i/%i : %s'%(n, len(f), descr)
#         parse = parser.raw_parse(descr)
#         alksdjfhladksjhflkajhsdf
#         treestr = get_str(parse.next())
#         treefile.write('%s\t%s\n'%(cid, treestr))
# with open(os.path.join(root, 'mapping'),'w') as mapFile:
#     cPickle.dump(mapping, mapFile)
qIN = Queue()
qOUT = Queue()
n_procs = 7
cids = seen_ids()
for n,line in enumerate(f):
    if not len(line):
        continue
    cid, descr = line.split('\t')
    if cid in cids:
        print 'Already seen %s'%(cid)
        continue
    qIN.put([cid, descr, n, len(f)])

for i in range(n_procs*4):
    qIN.put(None)
p = []
if __name__=='__main__':
    print 'starting'
    for i in range(n_procs):
        p.append(Process(target=worker, args=(qIN, qOUT)))
    for pr in p:
        pr.start()
    for pr in p:
        pr.join()
    print 'All are joined'
    with open(os.path.join(root, 'trees.txt'),'a') as treefile:
        while not qOUT.empty():
            cid, parse = qOUT.get()
            parse = Tree.fromstring(parse)
            treestr = get_str(parse)
            print 'Writing %s %s'%(cid, treestr)
            treefile.write('%s\t%s\n'%(cid, treestr))



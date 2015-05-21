# Accepts a document of the form:
#
# <img_id> (tab) <img_description> (return)
#
# and then generates a tree of each img_description, tokenizing the
# words as it goes. it maintains a persistent word tokenization
# dictionary, called mapping, so that you can call it on an arbitrary
# number of files and there will be no word collisions. The parse trees
# are also maintained in a continuous file.
#
#
# currently generates ALL trees for dependency and constituency
#
# note that the RAW tree files use a double \n\n as a separator
import os
import cPickle
from nltk.parse import stanford
from nltk.tree import Tree
from multiprocessing import Process, Queue
import time
import json

# You will need to customize this
if os.path.isdir('/afs/.ir.stanford.edu/users/n/d/ndufour'):
    pfx = '/afs/.ir.stanford.edu/users/n/d/ndufour/'
else:
    pfx = '/Users/ndufour/'
PARSER_LOC = pfx + 'stanford-parser/stanford-parser.jar'
MODEL_LOC = pfx + 'stanford-parser/stanford-parser-3.5.2-models.jar'
CONST_LOC = pfx + 'stanford-parser/stanford-parser-3.5.2-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
DEP_LOC = pfx + 'stanford-parser/stanford-parser-3.5.2-models/edu/stanford/nlp/models/parser/nndep/english_SD.gz'

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

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
        return mapping

def seen_ids():
    # returns a set of IDs tha thave already been seen by the parser
    if not os.path.exists(os.path.join(root, 'trees')):
        return set()
    else:
        sid = set()
        f = open(os.path.join(root, 'trees')).read().split('\n')
        for i in f:
            if not len(i):
                continue
            cid, _ = i.split('\t')
            sid.add(cid)
        return sid

mapping = get_map()
seen = seen_ids()

def init_parser(parseType='C'):
    # initializes the parser
    #
    # parseType == 'C' is a constituency tree (via stanford PCFG)
    # parseType == 'D' is a dependency tree (english_SD)
    os.environ['STANFORD_PARSER'] = PARSER_LOC
    os.environ['STANFORD_MODELS'] = MODEL_LOC
    if parseType == 'C':
        parser = stanford.StanfordParser(model_path=CONST_LOC)
    elif parseType == 'D':
        parser = stanford.StanfordParser(model_path=CONST_LOC)
    else:
        print 'Unrecognized parser type request'
        return
    return parser

def gk(item):
    if not mapping.has_key(item):
        with open(os.path.join(root, 'mapping'),'a') as mapf:
            v = item
            k = len(mapping)
            mapping[v] = k
            mapf.write('%s\t%i\n'%(v,k))
    return mapping[item]

def worker(qIN, qOUTC, qOUTD, workerN, nWorkers):
    # queue worker; instantiates its own stanford parse, then dequeues
    # sentences from qIN, parses them, adds them to qOUT.
    # format of each qIN item:
    # <id> <sentence> <num> <tot_num>
    parser_c = init_parser('C')
    parser_d = init_parser('D')
    for n,qr in enumerate(qIN):
        if (n%nWorkers)==workerN:
            cids, descs = qr
            print 'Worker %i starting parse chunk %i/%i...'%(workerN, n, len(qIN))
            parse = parser_c.raw_parse_sents(descs)
            print 'Worker %i constituency parse chunk %i/%i complete...'%(workerN, n, len(qIN))
            nparse = [y.next() for y in parse]
            for cid, parse in zip(cids, nparse):
                qOUTC.put([cid, str(parse)])
            parse = parser_d.raw_parse_sents(descs)
            print 'Worker %i dependency parse chunk %i/%i complete...'%(workerN, n, len(qIN))
            nparse = [y.next() for y in parse]
            for cid, parse in zip(cids, nparse):
                qOUTD.put([cid, str(parse)])

def get_str(tree):
    if type(tree)==unicode or type(tree)==str:
        return str(gk(tree.lower()))
    if len(tree) == 1:
        return get_str(tree[0])
    return '(' + ' '.join([get_str(x) for x in tree]) + ')'

def printTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%dh:%02dm:%02ds" % (h, m, s)

qIN = []
qOUTC = Queue()
qOUTD = Queue()
n_procs = 10
cids = seen_ids()
print 'Enqueuing sentences'
tot = 0
chunk_size = 2000
cur_chunk_desc = []
cur_chunk_id = []

# EXTRACT MS COCO DATA
if os.path.isdir('/afs/.ir.stanford.edu/users/n/d/ndufour'):
    filename1 = '/afs/.ir.stanford.edu/users/n/d/ndufour/imgcap/data/mscoco/captions_train2014.json'
    filename2 = '/afs/.ir.stanford.edu/users/n/d/ndufour/imgcap/data/mscoco/captions_val2014.json'
else:
    filename1 = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/mscoco/captions_train2014.json'
    filename2 = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/mscoco/captions_val2014.json'

f = []
f1 = open(filename1, 'r').read()
f+= json.loads(f1)['annotations']
f1 = open(filename2, 'r').read()
f+= json.loads(f1)['annotations']

for n,line in enumerate(f):
    cid, descr = [str(line['image_id']), line['caption'].decode('utf-8')]
    cid = 'mscoc_' + cid
    if cid in cids:
        print 'Already seen %s'%(cid)
        continue
    cur_chunk_desc.append(descr)
    cur_chunk_id.append(cid)
    if len(cur_chunk_desc)==chunk_size:
        qIN.append([cur_chunk_id, cur_chunk_desc])
        cur_chunk_desc = []
        cur_chunk_id = []
    tot += 1
qIN.append([cur_chunk_id, cur_chunk_desc])

# EXTRACT FLICKR DATA
if os.path.isdir('/afs/.ir.stanford.edu/users/n/d/ndufour'):
    filename = '/afs/.ir.stanford.edu/users/n/d/ndufour/imgcap/data/flickr_30k_v20130124.token'
else:
    filename = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/flickr_30k_v20130124.token'
f = open(filename, 'r').read().split('\n')
for n,line in enumerate(f):
    if not len(line):
        continue
    cid, descr = line.decode('utf-8').split('\t')
    cid = 'flickr_' + cid.split('#')[0]
    if cid in cids:
        print 'Already seen %s'%(cid)
        continue
    cur_chunk_desc.append(descr)
    cur_chunk_id.append(cid)
    if len(cur_chunk_desc)==chunk_size:
        qIN.append([cur_chunk_id, cur_chunk_desc])
        cur_chunk_desc = []
        cur_chunk_id = []
    tot += 1
qIN.append([cur_chunk_id, cur_chunk_desc])
p = []

if __name__=='__main__':
    print 'starting'
    for i in range(n_procs):
        p.append(Process(target=worker, args=(qIN, qOUTC, qOUTD, i, n_procs)))
    for pr in p:
        pr.start()
        time.sleep(1)
    curcnt = 0
    start = time.time()
    with open(os.path.join(root, 'trees_const'),'a') as tree_const, open(os.path.join(root, 'trees_const_raw'), 'a') as tree_const_raw, open(os.path.join(root, 'trees_dep'),'a') as tree_dep, open(os.path.join(root, 'trees_dep_raw'), 'a') as tree_dep_raw:
        while curcnt != tot or not qOUTC.empty() or not qOUTD.empty():
            curcnt += 1
            cid, parse = qOUTC.get()
            tree_const_raw.write('%s\t%s\n'%(cid, parse.replace('\n','')))
            parse = Tree.fromstring(parse)
            treestr = get_str(parse)
            remtime = printTime((tot-(curcnt+1))*(time.time()-start)/(curcnt+1))
            tree_const.write('%s\t%s\n'%(cid, treestr))
            cid, parse = qOUTD.get()
            tree_dep_raw.write('%s\t%s\n'%(cid, parse.replace('\n','')))
            parse = Tree.fromstring(parse)
            treestr = get_str(parse)
            remtime = printTime((tot-(curcnt+1))*(time.time()-start)/(curcnt+1))
            if not curcnt % (chunk_size*n_procs):
                print '(%i / %i) [rem: %s] %s %s'%(curcnt, tot, remtime, cid, treestr)
            tree_dep.write('%s\t%s\n'%(cid, treestr))
    for pr in p:
        pr.join()
    print 'All are joined'



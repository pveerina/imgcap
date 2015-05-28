from collections import Counter
from collections import defaultdict as ddict
import os
import json
from nltk import Tree

need = ddict(lambda: [])
with open('to_parse','r') as f:
    f = f.read().split('\n')
    for line in f:
        if not len(line):
            continue
        cid, sent = line.split('\t')
        need[cid].append(sent)

need_ids = set(need.keys())
raw_const = ddict(lambda: [])
with open('has_dups/trees_const_raw','r') as f, open('trees_const_raw','w') as f2:
    f = f.read().split('\n')
    for n,line in enumerate(f):
        print 'const: %i/%i'%(n, len(f))
        if not len(line):
            continue
        cid, tree = line.split('\t')
        if not cid in need_ids:
            continue
        tree = Tree.fromstring(tree)
        raw_const[cid].append(' '.join(tree.leaves()))
        f2.write('%s\n'%line)

raw_dep = ddict(lambda: [])
with open('has_dups/trees_dep_raw','r') as f, open('trees_dep_raw','w') as f2:
    f = f.read().split('\n')
    for n,line in enumerate(f):
        print 'dep: %i/%i'%(n, len(f))
        if not len(line):
            continue
        cid, tree = line.split('\t')
        if not cid in need_ids:
            continue
        tree = Tree.fromstring(tree)
        raw_dep[cid].append(' '.join(tree.leaves()))
        f2.write('%s\n'%line)

with open('has_dups/trees_dep','r') as f, open('trees_dep','w') as f2:
    f = f.read().split('\n')
    for n,line in enumerate(f):
        print 'dep: %i/%i'%(n, len(f))
        if not len(line):
            continue
        cid, tree = line.split('\t')
        if not cid in need_ids:
            continue
        f2.write('%s\n'%line)

with open('has_dups/trees_const','r') as f, open('trees_const','w') as f2:
    f = f.read().split('\n')
    for n,line in enumerate(f):
        print 'const: %i/%i'%(n, len(f))
        if not len(line):
            continue
        cid, tree = line.split('\t')
        if not cid in need_ids:
            continue
        f2.write('%s\n'%line)


bad_dep = 0
bad_const = 0
bad_keys = set()
rdf = dict()
rcf = dict()
nf = dict()
for k in need:
    rdf = [x.replace(' ','') for x in raw_dep[k]]
    rcf = [x.replace(' ','') for x in raw_const[k]]
    nf = [x.replace(' ','') for x in need[k]]
    for j in nf:
        if j not in rdf:
            bad_dep+=1
            print 'Need dep: %s %s'%(k,j)
            bad_keys.add(k)
            break
        if j not in rcf:
            bad_const+=1
            print 'Need const: %s %s'%(k,j)
            bad_keys.add(k)
            break
bad_keys = list(bad_keys)

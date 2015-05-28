import enchant
import os
import glob

import os
from collections import Counter
from collections import defaultdict as ddict
from nltk.tokenize import wordpunct_tokenize as wt
import string

mroot = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'
gfile = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/data/glove.840B.300d.txt'

pfile = os.path.join(mroot, 'to_parse')
nfile = os.path.join(mroot, 'to_parse_prep')

our_words = Counter()
our_words_lower = ddict(lambda: set())



extant = open(pfile,'r').read().strip().split('\n')
extant = [x.split('\t') for x in extant]
exclude = string.punctuation
retain = ',.&\''
new = []
for n, x in enumerate(extant):
    print n
    #print '%i/%i'%(n, len(extant))
    # tokenize
    sent = x[1]
    # remove duplicate punctuation marks
    for i in exclude:
        while i+i in sent:
            sent = sent.replace(i+i,i)
        if i in sent:
            if i in retain:
                sent = i.join(sent.split(i))
                sent = i.join(sent.split(' '+i))
            else:
                sent = ' '.join(sent.split(i))
    sent = ' '.join(sent.split())
    tok = wt(sent)
    for w in tok:
        our_words[w] += 1
        our_words_lower[w.lower()].add(w)
    sent = sent.capitalize()
    if not sent[-1] == '.':
        sent += '.'
    new.append([x[0], sent])

normalized = ddict(lambda:'')
for i in our_words_lower.keys():
    mxw = ''
    mxc = 0
    for j in our_words_lower[i]:
        if our_words[j] > mxc:
            mxw = j
            mxc = our_words[j]
    normalized[i] = mxw

for n, x in enumerate(new):
    print n
    os = x[1]
    sent = x[1]
    sent = sent.capitalize()
    tok = wt(sent)
    sent = ' '.join([tok[0]] + [normalized[z] for z in tok[1:]])
    sent = ' '.join(sent.split())
    for i in exclude:
        sent = sent.replace(' '+i, i)
    x[1] = sent

newnew = ['%s\t%s'%(x[0],x[1]) for x in new]

with open(nfile, 'w') as f:
    f.write('\n'.join(newnew))

import xml.etree.ElementTree as et
from glob import glob
from os.path import join as opj
mroot = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'
idfls = glob(opj(mroot, 'parse_ids','*'))
parsefls = glob(opj(mroot, 'parse_out','*.xml'))
dest = opj(mroot, 'dep_parse_prep')

towrite = []
for n,(parsefl,idfl) in enumerate(zip(parsefls, idfls)):
    print '%i/%i %s'%(n, len(parsefls), parsefl)
    p = et.parse(parsefl)
    root = p.getroot()
    sents = root.getchildren()[0].getchildren()[0].getchildren()
    ids = open(idfl, 'r').read().strip().split('\n')
    if len(sents) != len(ids):
        print 'ERROR!'
    for cid,s in enumerate(zip(ids,sents)):
        tree = []
        parse = s[1].getchildren()[1]
        for pz in parse:
            tuple1 = (pz[0].text, int(pz[0].attrib['idx']))
            tuple2 = (pz[1].text, int(pz[1].attrib['idx']))
            tree.append((tuple1, tuple2))
        towrite.append('%s\t%s'%(cid, str(tree)))

with open(dest,'w') as f:
    f.write('\n'.join(towrite))

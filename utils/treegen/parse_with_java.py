# we need to split this up into files
import os

os.mkdir('parse_in')
os.mkdir('parse_out')
os.mkdir('parse_ids')

mroot = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'
roots = [os.path.join(mroot, x) for x in ['parse_in','parse_out','parse_ids']]

file_cnt = 0
abs_cnt = 0
chk_sz = 200
nparselists = 0

#parselist = [open(os.path.join(mroot,'parselist_%i'%x),'w') for x in range(5)]
parselist = open(os.path.join(mroot,'parselist'),'w')
to_parse = open('to_parse_prep','r').read().split('\n')

fname = 'file_%4i'%file_cnt
towrite = ''
for n, i in enumerate(to_parse):
    if not len(i):
        continue
    if not n%chk_sz:
        print file_cnt
        if len(towrite):
            f.write(towrite.strip())
        towrite = ''
        fname = 'file_%04i'%file_cnt
        f = open(os.path.join(roots[0], fname),'w')
        f2 = open(os.path.join(roots[2], fname),'w')
        #parselist[file_cnt%nparselists].write('%s\n'%os.path.join(roots[0], fname))
        parselist.write('%s\n'%os.path.join(roots[0], fname))
        file_cnt += 1
    cid, csent = i.split('\t')
    towrite += '%s\n'%csent
    f2.write('%s\n'%cid)
if len(towrite):
    f.write(towrite.strip())
f.close()
f2.close()
parselist.close()
#_ = [x.close() for x in parselist]

# java -cp stanford-corenlp-3.5.2.jar:stanford-corenlp-3.5.2-models.jar:xom.jar:joda-time.jar:jollyday.jar:ejml-0.23.jar -Xmx8g edu.stanford.nlp.pipeline.StanfordCoreNLP -ssplit.eolonly -annotators "tokenize,ssplit,pos,depparse" -filelist /Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen/parselist -outputDirectory /Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen/parse_out

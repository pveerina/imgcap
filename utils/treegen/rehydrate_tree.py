# exports a tree structure for a parse

import os

try:
    root = os.path.dirname(os.path.abspath(__file__))
except:
    root = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'
#root = '/Users/ndufour/Dropbox/Class/CS224D/project/imgcap/utils/treegen'
def get_imap():
    # returns the current mapping
    if not os.path.exists(os.path.join(root, 'mapping')):
        return dict()
    else:
        f = open(os.path.join(root, 'mapping'),'r').read()
        mapping = dict()
        for i in f.split('\n'):
            if len(i):
                spl = i.split('\t')
                mapping[int(spl[1])] = spl[0]
        return mapping

class treeGet():
    def __init__(self):
        self.map = get_imap()
    def get(self, strRep):
        rep = eval(strRep.replace('(','[').replace(')',']').replace(' ',','))
        return treeStruct(rep, self.map)

class treeStruct():
    def __init__(self, rep, mapping):
        rep = eval(strRep.replace('(','[').replace(')',']').replace(' ',','))
        self.nodes = []
        self.root = node(None, rep, mapping, self)
    def prettyPrint(self):
        def rPrettyPrint(node, cstr):
            if node.isLeaf:
                print cstr[:-1] + '|' + '----' + node.word
            else:
                print cstr[:-1] + '|' + '-----'
                cstr += '    |'
                for c in node.children:
                    print cstr + '\n' + cstr
                    if c == node.children[-1]:
                        cstr = cstr[:-1] + ' '
                    rPrettyPrint(c, cstr)
        rPrettyPrint(self.root, '')
    def toString(self):
        def rts(node, cstr):
            if node.isLeaf:
                cstr += ' ' + node.word
                return cstr
            else:
                for c in node.children:
                    cstr += rts(c, cstr)
                return cstr
        return rts(self.root, '').strip()

class node():
    def __init__(self, parent, rep, mapping, tree):
        self.parent = parent
        tree.nodes.append(self)
        if type(rep) == int:
            self.isLeaf = True
            self.ident = rep
            self.children = None
            self.label = 0
            self.word = mapping[rep]
        else:
            self.isLeaf = False
            self.ident = None
            self.label = len(rep)
            self.word = None
            self.children = []
            for i in rep:
                self.children.append(node(self, i, mapping, tree))

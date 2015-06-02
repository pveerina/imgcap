# exports a class that will elaborate data into trees
import collections
UNK = 'UUUNKKK'

class Node: # a node in the tree
    def __init__(self, wordidx, glob_idx):
        self.word = wordidx # the index
        self.parent = None
        self.left = []
        self.right = []
        self.isLeaf = True
        self.fprop = False
        self.idx = 0
        self.glob_idx = glob_idx
        self.o = None
        self.i = None
        self.u = None
        self.l = None
        self.r = None
	self.numLeft = None
	self.numRight = None

class Tree:
    def __init__(self, tree, img=None, num=None):
        # img = the image label
        # num = number of this caption
        #
        # recieves the tree as a list of tuples.
        # such that tree[i] = ith dependency tuple
        # the dependency tuple is:
        #   (x, xi, y, yi)
        # x: governor word index (parent)
        # y: dependent word index (child)
        # xi: index of x in the sentence
        # yi: index of y in the sentence
        if type(tree) == str:
            tree = eval(tree) # on event that it wasn't passed as a list
        nodes = dict()
        for p, pi, c, ci in tree[1:]:
            # so we'll see left children in reverse order and right children
            # in the incorrect order.
            if not nodes.has_key(pi):
                nodes[pi] = Node(p, pi)
            nodes[pi].isLeaf = False
            if not nodes.has_key(ci):
                nodes[ci] = Node(c, ci)
            nodes[ci].parent = nodes[pi]
            if pi > ci:
                # it's a left child
                for node in nodes[pi].left:
                    node.idx += 1 # increment each one of them, since left nodes are added in reverse order
                nodes[pi].left.append(nodes[ci])
            else:
                # it's a left child
                nodes[ci].idx = len(nodes[pi].right)
                nodes[pi].right.append(nodes[ci])
        # recall that left children are added in reverse order, so iterate
        # over the nodes and reverse all their right children
        for node in nodes.values():
            node.left = node.left[::-1]
        self.root = nodes[tree[0][-1]]
        self.img = img
        self.num = num

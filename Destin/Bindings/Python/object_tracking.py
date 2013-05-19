# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:45:29 2013

@author: ted
"""

import common as cm

cm.init(centroids=[4,8,32,32,64,32,4],
        video_file="/home/ted/Dropbox/destin/moving_square.mov",
        learn_rate=0.05)
                 
cm.video_source.enableDisplayWindow()
cm.network.setIsPOSTraining(True)

def callback(iter):
    cm.printStats()
    #cm.network.printBeliefGraph(cm.top_layer, 0, 0)
    print "iter:",iter
    cm.printFPS()

def showTree(index):
    if index >= tm.getMinedTreeCount():
        print "out of bounds"
        return
    tm.displayMinedTree(index)
    tm.printMinedTreeStructure(index)
    cm.wk()
    
def doTracking(delay=0):
    for i in xrange(200):
        cm.go(1)
        m = cm.video_source.getOutputColorMat()
        for j in xrange(tm.getFoundSubtreeCount()):
            tm.displayFoundSubtreeBorders(j, m, False, 2, 0)
        #cm.wk(delay)
    
cm.the_callback = callback

n = cm.network

if True:
    cm.go(3200)
    n.save("ot.dst")
else:
    n.load("ot.dst")

#cm.network.save("ot.dst")
tm = cm.pd.DestinTreeManager(n,2)

cm.freezeTraining()
for i in xrange(100):
    cm.go(1)
    tm.addTree()

support = 30
print "mining..."
found = tm.mine(support)
print "Found", found


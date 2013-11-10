# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:45:29 2013

@author: ted
"""

import common as cm

#layer_widths=[16, 8, 7, 6, 5, 4, 3, 2, 1]
#centroids=   [2, 32,32,64,16,16,16,16,4]
layer_widths=[16, 8, 7, 6, 5, 4, 3, 2, 1]
centroids=   [2,  16,16,16,16,16, 16, 16, 4]
#centroids=[8,32,32,64,32,4]


isTraining=False # train from scratch, or reload from previous run

import os
print "pid is:" + str(os.getpid())

cm.init(centroids=centroids,
        video_file="moving_square.avi",
        learn_rate=0.05,
        layer_widths=layer_widths,
        img_width=256
        )
#

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

if isTraining:
    cm.go(3200)
    n.save("ot.dst")
else:
    n.load("ot.dst")

#cm.network.save("ot.dst")
tm = cm.pd.DestinTreeManager(n,3)
print "size of winning tree is: " + str(tm.getWinningCentroidTreeSize())
cm.freezeTraining()
for i in xrange(100):
    cm.go(1)
    tm.addTree()

support = 10
print "mining..."
found = tm.mine(support)
print "Found", found

doTracking()
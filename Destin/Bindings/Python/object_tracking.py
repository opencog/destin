# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:45:29 2013

@author: ted
"""

import common as cm
import pydestin as pd
# load the layer_widths and centroids from the 
# object_tracking_config.py
from object_tracking_config import *

isTraining=True # train from scratch, or reload from previous run

cm.init(centroids=centroids,
        video_file="moving_square.avi",
        learn_rate=0.05,
        layer_widths=layer_widths,
        img_width=256
        )
#

cm.video_source.enableDisplayWindow()
cm.network.setIsPOSTraining(True)

write_video=True

video_writer = pd.VideoWriter("obtracking_output.avi",15)


callback_iters=0
def callback(iter):
    #cm.network.printBeliefGraph(cm.top_layer, 0, 0)
    if iter %20 == 0:
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
        if write_video:
                video_writer.write(m)
                
        if delay != -1:        
            cm.wk(delay)
    
cm.the_callback = callback

n = cm.network

if isTraining:
    cm.go(3200)
    n.save("ot.dst")
else:
    n.load("ot.dst")

#cm.network.save("ot.dst")
tm = cm.pd.DestinTreeManager(n,1)
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

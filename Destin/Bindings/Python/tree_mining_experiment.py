# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:14:05 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv


## Init destin
centroids = [2, 4, 4, 16, 8, 8, 4, 4]
dn = pd.DestinNetworkAlt(pd.W512, 8, centroids, True)
dn.setBeliefTransform(pd.DST_BT_P_NORM)
temperatures = [2,2,2,2,2,2,2,2]
dn.setTemperatures(temperatures)
dn.setIsPOSTraining(True)

## load video
vs = pd.VideoSource(False, "moving_circle.avi")
vs.enableDisplayWindow()

## setup tree manager
bottom_layer = 5
tm = pd.DestinTreeManager(dn, bottom_layer)

def train():
    for i in range(200):
        if not vs.grab():
            continue
        
        if(i % 10 == 0):
            print "Iteration " + str(i)
            
        dn.doDestin(vs.getOutput())
    
    
def getTrees():
    for i in range(50):
        if not vs.grab():
            print "could not grab frame"
            continue;
            
        dn.doDestin(vs.getOutput())
        tm.addTree()
        
def displayMinedTree(index):
    if index >= found:
        print "out of bounds"
        return
    tm.displayMinedTree(index)
    cv.WaitKey(200)
    
#train()
#dn.save("minedtree.dst")
dn.load("minedtree.dst")
getTrees()

support = 10
found = tm.mine(support)
print "found %i trees" % (found)

displayMinedTree(0)




    
    
    

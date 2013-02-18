# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:14:05 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv


## Init destin
centroids = [2, 2, 4, 4, 4, 16, 8, 8]
dn = pd.DestinNetworkAlt(pd.W512, 8, centroids, True)
dn.setBeliefTransform(pd.DST_BT_P_NORM)
#dn.setBeliefTransform(pd.DST_BT_NONE)

uniform_temp = 2.0
temperatures = []
for i in range(8):
    temperatures.append(uniform_temp)

dn.setTemperatures(temperatures)
dn.setIsPOSTraining(True)

dn.setCentImgWeightExponent(4)

## load video
vs = pd.VideoSource(False, "movingX.avi")
vs.enableDisplayWindow()

## setup tree manager
bottom_layer = 5
tm = pd.DestinTreeManager(dn, bottom_layer)

train_frames = 600

def train():
    for i in range(train_frames):
        if not vs.grab():
            continue
        
        dn.doDestin(vs.getOutput())
        dn.printBeliefGraph(7,0,0)
            
def getTrees():
    dn.setIsPOSTraining(False) # freeze training
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

def displayCentroidImages(layer):
    dn.displayLayerCentroidImages(layer)
    cv.WaitKey(200)
    
      
    
train()
dn.save("minedtree.dst")
#dn.load("minedtree.dst")
getTrees()

support = 10
found = tm.mine(support)
print "found %i trees" % (found)

displayMinedTree(0)




    
    
    

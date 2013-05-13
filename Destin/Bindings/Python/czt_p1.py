# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:50:00 2013

@author: teaera
"""

import os, errno
import cv2.cv as cv
import time
import pydestin as pd

ims = pd.ImageSouceImpl()

'''
nLayer = 8
centroids = [6,8,16,32,64,32,16,8]
isUniform = True
size = 512*512
extRatio = 1
network = pd.DestinNetworkAlt(pd.W512, nLayer, centroids, isUniform)
network.reinitNetwork_c1(pd.W512, nLayer, centroids, isUniform, size, extRatio)
network.setFixedLearnRate(.1)
network.setBeliefTransform(pd.DST_BT_NONE)
'''

#############################################################################
def dcis(layer):
    network.displayLayerCentroidImages(layer,1000)
    cv.WaitKey(100)

def load_ims_fld(ims, fld):
    if not fld.endswith("/"):
        fld += "/"
    for each in os.listdir(fld):
        ims.addImage(fld + each)

def init_destin(siw=pd.W512, nLayer=8, centroids=[4,8,16,32,64,32,16,8],
                isUniform=True, size=512*512, extRatio=1):
    temp_network = pd.DestinNetworkAlt(pd.W512, nLayer, centroids, isUniform)
    temp_network.reinitNetwork_c1(pd.W512, nLayer, centroids, isUniform, size, extRatio)
    temp_network.setFixedLearnRate(.1)
    temp_network.setBeliefTransform(pd.DST_BT_NONE)
    return temp_network
#############################################################################

print("------")

'''
network = init_destin()
#
vs1 = pd.VideoSource(False, "/home/teaera/destin_ted_temp/Destin/Misc/ABCD.avi")
#vs1 = pd.VideoSource(True, "", 0)
#vs2 = pd.VideoSource(True, "", 0+1)
vs1.enableDisplayWindow_c1("left")
#vs2.enableDisplayWindow_c1("right")
vs1.grab()
#vs2.grab()

maxCount = 16000
currCount = 0
t1 = pd.Transporter()
while(vs1.grab()):
#while(vs1.grab() and vs2.grab()):
    currCount += 1
    if currCount <= maxCount:
        if currCount % 10 == 0:
            print "Iteration " + str(currCount)
        t1.setSource(vs1.getOutput())
        t1.transport()
        network.doDestin_c1(t1.getDest())
        #network.doDestin(t1.getDest())
    else:
        break
'''

'''
'''
network = init_destin()
#network = init_destin(centroids=[8,16,32,64,32,32,32,16])
#load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.6/1")
load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.7/1")
maxCount = 1000

for i in range(maxCount):
    if i % 10 == 0:
        print "Iteration " + str(i)
    ims.findNextImage()
    f = ims.getGrayImageFloat()    
    network.doDestin_c1(f)
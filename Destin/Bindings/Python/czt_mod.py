# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:49:09 2013

@author: teaera
"""

import os
import cv2.cv as cv
import pydestin as pd

cl = pd.czt_lib()
cl2 = pd.czt_lib2()

#############################################################################
'''
Display centroids images!
'''
def dcis(network, layer):
    network.displayLayerCentroidImages(layer,1000)
    cv.WaitKey(100)

'''
Save centroids images!
'''
def saveCens(network, layer, saveLoc):
    network.saveLayerCentroidImages(layer, saveLoc)

'''
Load images in one folder into an 'ims'!!!
'''
def load_ims_fld(ims, fld):
    if not fld.endswith("/"):
        fld += "/"
    for each in os.listdir(fld):
        ims.addImage(fld + each)

'''
Used to init DeSTIN, but compatible by setting 'extRatio'!
'''
def init_destin(siw=pd.W512, nLayer=8, centroids=[4,8,16,32,64,32,16,8],
                isUniform=True, size=512*512, extRatio=1, isExtend=False):
    if isExtend:
        temp_network = pd.DestinNetworkAlt(pd.W512, nLayer, centroids, isUniform,
                                           isExtend, size, extRatio)
    else:
        temp_network = pd.DestinNetworkAlt(pd.W512, nLayer, centroids, isUniform)
    temp_network.setBeliefTransform(pd.DST_BT_NONE)
    return temp_network

'''
Use czt_lib to format images in one folder to the size 512*512 and store
into another folder!
'''
def processFld(inFld, outFld):
    if not inFld.endswith("/"):
        inFld += "/"
    if not outFld.endswith("/"):
        outFld += "/"
    for each in os.listdir(inFld):
        cl.isNeedResize(inFld + each)
        cl.write_file(outFld + each)

'''
Use the existing network and ims to train!
Default number is 16,000.
'''
def train_ims(network, ims, maxCount=16000):
    for i in range(maxCount):
        if i % 10 == 0:
            print "Iteration " + str(i)
        ims.findNextImage()
        f = ims.getGrayImageFloat()    
        network.doDestin_c1(f)

'''
Use one folder as input, and use another folder as additional info!
'''
def train_2flds(network, fld1, fld2, repeatCount=1600):
    if not fld1.endswith("/"):
        fld1 += "/"
    if not fld2.endswith("/"):
        fld2 += "/"
    for i in range(repeatCount):
        if i % 10 == 0:
            print "RepeatTime: " + str(i)
        for each in os.listdir(fld1):
            f = cl.combineImgs(fld1+each, fld2+each)
            network.doDestin_c1(f)

#############################################################################
# Testing functions:
def train_ims_randomInfo(network, ims, size, extRatio, maxCount=16000):
    for i in range(maxCount):
        if i % 10 == 0:
            print "Iteration " + str(i)
        ims.findNextImage()
        f1 = ims.getGrayImageFloat()
        f2 = cl2.getFloatArr(size*extRatio)
        cl2.combineInfo_extRatio(f1, size, extRatio, f2)
        network.doDestin_c1(f2)

def train_only(network, tempIn, maxCount=16000):
    for i in range(maxCount):
        if i % 10 == 0:
            print "Iteration " + str(i)
        network.doDestin_c1(tempIn)

#############################################################################
def drawCurve(inFile, times):
    fCont = open(inFile).read().split("\n")[:times]
    quality = {}
    for i in range(8):
        quality[str(i)] = []
    for i in range(times):
        lCont = fCont[i].split("  ")[:8]
        for i in range(8):
            quality[str(i)].append(float(lCont[i]))
    import matplotlib.pyplot as plt
    for i in range(8):
        plt.figure()
        plt.plot(quality[str(i)], "r*-")
        plt.savefig("/home/teaera/Pictures/2013.7.4_"+str(i)+".jpg")
    plt.show()

#drawCurve('/home/teaera/destin_ted_temp/Destin/1', 50)


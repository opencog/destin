# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:49:09 2013

@author: teaera
"""

import os
import cv2.cv as cv
import pydestin as pd

#cl = pd.czt_lib()
cm = pd.CztMod()

#############################################################################
"""
Save the current user's home folder.
"""
homeFld = os.getenv("HOME")

if not homeFld:
    homeFld = os.getenv("USERPROFILE")

"""
Display centroids images!
"""
def dcis(network, layer):
    network.displayLayerCentroidImages(layer,1000)
    cv.WaitKey(100)

"""
Save centroids images!
"""
def saveCens(network, layer, saveLoc):
    network.saveLayerCentroidImages(layer, saveLoc)

"""
Load images in one folder into an 'ims'!!!
"""
def load_ims_fld(ims, fld):
    if not fld.endswith("/"):
        fld += "/"
    for each in os.listdir(fld):
        ims.addImage(fld + each)

"""
Used to init DeSTIN, but compatible by setting 'extRatio'!
"""
def init_destin(siw=pd.W512, nLayer=8, centroids=[4,8,16,32,64,32,16,8],
                isUniform=True, imageMode=pd.DST_IMG_MODE_GRAYSCALE):
    temp_network = pd.DestinNetworkAlt(siw, nLayer, centroids, isUniform, imageMode)
    #temp_network.setBeliefTransform(pd.DST_BT_NONE)
    
    return temp_network

"""
Use the existing network and ims to train!
Default number is 16,000.
"""
def train_ims(network, ims, maxCount=16000):
    for i in range(maxCount):
        if i % 10 == 0:
            print "Iteration " + str(i)
        ims.findNextImage()
        f = ims.getGrayImageFloat()    
        network.doDestin(f)

"""
Use one folder as input, and use another folder as additional info!
"""
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
            network.doDestin(f)

"""
Get the time stamp for today
"""
import datetime
def getTimeStamp():
    now = datetime.datetime.now()
    return str(now.year) + "." + str(now.month) + "." + str(now.day)

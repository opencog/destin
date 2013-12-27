# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:55:26 2013

@author: teaera
"""

import os
import time
import cv2.cv as cv
import pydestin as pd

import czt_mod as cm

#############################################################################
# Pre-processing
#
#cm.processFld(cm.homeFld + "/Work/RECORD/2013.5.8/org", cm.homeFld + "/Work/RECORD/2013.5.8/pro_1")

#############################################################################
# 2013.5.7
'''
dn = cm.init_destin(extRatio=1)
ims = pd.ImageSouceImpl(512, 512)
#cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_2")
#cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_2_2")
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013_5_8_1.jpg")
'''


#############################################################################
# 2013.5.9
'''
start_time = time.time()
dn = cm.init_destin(extRatio=2)
cm.train_2flds(dn, cm.homeFld + "/Work/RECORD/2013.5.8/pro_3", cm.homeFld + "/Work/RECORD/2013.5.8/pro_add_3", 10)
end_time = time.time()
print("Cost: %s secs!" % (str(end_time-start_time)))
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.10
'''
size = 512*512
extRatio = 1
dn = cm.init_destin(extRatio=extRatio)
cm.cl.isNeedResize(cm.homeFld + "/Work/RECORD/2013.5.8/pro_add_3/1.jpg")
f = cm.cl.get_float512()
cm.train_only(dn, f, 3000)
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.13
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl(512, 512)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=5000)
for i in range(8):
    #cm.saveCens(dn, i, cm.homeFld + "/Pictures/segmentation_face/%s.jpg"%(str(i)))
    cm.saveCens(dn, i, cm.homeFld + "/Pictures/original_face/%s.jpg"%(str(i)))
'''

#############################################################################
# 2013.5.15
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl(512, 512)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=3500)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.15_fixed.jpg")
#cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.15_decay.jpg")
'''

#############################################################################
# 2013.5.15 centroids=[8,16,32,64,32,16,8,4]
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl(512, 512)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_2_2")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.15_decay_2.jpg")
'''

#############################################################################
# 2013.5.16
'''
dn = cm.init_destin(centroids=[8,16,32,64,64,64,32,16])
ims = pd.ImageSouceImpl(512, 512)
#cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_4")
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.8/pro_5")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.16_1.jpg")
cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.16_2.jpg")
'''

#############################################################################
# Pre-processing
# 2013.5.20 I have finished the import of Shuo's codes to do pre-processing!
#
'''
>>>PART 1. About the original faces:
'''
'''
# STEP 1. Extract from every folder!
inFile = cm.homeFld + "/Downloads/orl_faces/s%s/1.pgm"
outFile = cm.homeFld + "/Work/EclipseWorkSpace/Test1_c1/images_c1/%s.pgm"
for i in range(40):
    os.system("cp " + inFile%(str(i+1)) + " " + outFile%(str(i+1)))
#
# STEP 2. Processing
cm.processFld(cm.homeFld + "/Work/RECORD/2013.5.20/org_0", cm.homeFld + "/Work/RECORD/2013.5.20/pro_0")
'''
#------#
'''
>>>PART 2. About the generated results!
'''
'''
# Manually copied from out_1 to org_1...
# STEP 1. 
inFld = cm.homeFld + "/Work/RECORD/2013.5.20/org_1/"
files = os.listdir(inFld)
files.sort()
i = 1
for each in files:
    os.system("mv '" + inFld+each + "' " + inFld+str(i)+".jpg") # Because space, should use ''!!!
    i += 1
#
# STEP 2.
cm.processFld(cm.homeFld + "/Work/RECORD/2013.5.20/org_1", cm.homeFld + "/Work/RECORD/2013.5.20/pro_1")
'''

#############################################################################
# 2013.5.20
'''
#dn = cm.init_destin(centroids=[12,18,24,30,30,24,18,12])
dn = cm.init_destin()
ims = pd.ImageSouceImpl(512, 512)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.20/pro_2/")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.21
# STEP 1.
# Transform the downloaded images to gray-scale images first;
'''
cm.processFld(cm.homeFld + "/Work/RECORD/2013.5.21/downloads", cm.homeFld + "/Work/RECORD/2013.5.21/downloads_pro_mine")
'''

# STEP 2.
# Use Shuo's codes to get segmentated images;
# Eclipse;

#############################################################################
# 2013.5.21, 2013.5.22, 2013.5.23, 2013.5.24,
# STEP 3.
# Train original images;
'''
numImg = 2
centroids=[4,8,16,32,32,16,4,1]
for i in range(8):
    centroids[i] *= numImg
dn = cm.init_destin(centroids=centroids)
ims = pd.ImageSouceImpl(512, 512)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.5.21/z_2")
cm.train_ims(dn, ims, maxCount=2000)
cm.dcis(dn, 7)
cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.5.24_13.jpg")
'''

#############################################################################
# 2013.6.14
# I want to use matplotlib to draw the curve of validity!
# 2013.6.27
# Draw intra and inter seperately!
import matplotlib.pyplot as plt
'''
centroids = [6,8,10,12,12,8,6,4]
size = 512*512
extRatio = 1
isUniform = True
dn = cm.init_destin(centroids=centroids, extRatio=extRatio)
ims = pd.ImageSouceImpl(512, 512)
ims.addImage(cm.homeFld + "/Work/RECORD/2013.5.8/pro_1/3.jpg")
#cl2 = pd.czt_lib2()
#fOut = cl2.createFloatArr(size)
valDict = {}
intraDict = {}
interDict = {}
for i in range(8):
    valDict[str(i)] = []
    intraDict[str(i)] = []
    interDict[str(i)] = []
#maxCount = 3000
maxCount = 1000
for i in range(1, maxCount+1):
    if i % 10 == 0:
        print "Iteration " + str(i)
        for j in range(8):
            valDict[str(j)].append(dn.getValidity(j))
            intraDict[str(j)].append(dn.getIntra(j))
            interDict[str(j)].append(dn.getInter(j))
    ims.findNextImage()
    f = ims.getGrayImageFloat()
    #cl2.combineInfo_extRatio(f, size, extRatio, fOut)    
    dn.doDestin(f)
#
for i in range(8):
    plt.figure()
    plt.plot(range(len(valDict[str(i)])), valDict[str(i)], "r*-")
    plt.savefig(cm.homeFld + "/Pictures/2013.6.18_"+str(i)+".jpg")
plt.show()
'''

#############################################################################
# 2013.7.4
# Keep how to use 2-webcam as input;
'''
network = init_destin()
#
vs1 = pd.VideoSource(False, cm.homeFld + "/destin_ted_temp/Destin/Misc/ABCD.avi")
#vs1 = pd.VideoSource(True, "", 0)
#vs2 = pd.VideoSource(True, "", 0+1)
vs1.enableDisplayWindow("left")
#vs2.enableDisplayWindow("right")
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
        network.doDestin(t1.getDest())
        #network.doDestin(t1.getDest())
    else:
        break
'''

#############################################################################
# 2013.7.5, 2013.7.8
# After I merged the changes in DeSTIN, the mod should be changed;
'''
nLayer = 8
centroids = [4,8,16,32,32,16,8,4]
#centroids = [8,16,32,64,64,32,12,4]
#centroids = [4,8,16,32,32,16,8,4]
#centroids = [8,16,16,32,32,16,8,4]
#centroids = [8,16,32,64,64,32,12,4]
#centroids = [8,16,32,64,64,32,12,6]
#centroids = [8,16,32,64,72,36,18,6]
#centroids = [16,32,48,72,96,48,24,6]
#centroids = [32,48,64,72,96,72,24,6]
#centroids = [40,60,80,90,100,90,24,6]
#centroids = [4,4,4,4,4,4,4,6]
size = 512*512
#dn = cm.init_destin(centroids=centroids)
dn = cm.init_destin(centroids=centroids)

cl2 = pd.czt_lib2()

ims = pd.ImageSouceImpl(512, 512)
# Test just ONE image
#ims.addImage(cm.homeFld + "/Work/RECORD/2013.7.8/test1/3.jpg")
# Test FOUR images from the folder
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.7.8/test1/")
# TEST SIX images
#cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.7.8/test2/")

sepDict = {}
varDict = {}
quaDict = {}
for i in range(nLayer):
    sepDict[str(i)] = []
    varDict[str(i)] = []
    quaDict[str(i)] = []

maxCount = 1500
for i in range(1, maxCount+1):
    if i % 10 == 0:
        print "Iteration " + str(i)
        for i in range(nLayer):
            sep = dn.getSep(i)
            var = dn.getVar(i)
            qua = dn.getQuality(i)
            sepDict[str(i)].append(sep)
            varDict[str(i)].append(var)
            quaDict[str(i)].append(qua)
    ims.findNextImage()
    f = ims.getGrayImageFloat()
    dn.doDestin(f)

cm.dcis(dn, 7)
cm.saveCens(dn, 7, cm.homeFld + "/Pictures/2013.7.8_layer7.jpg")
for i in range(nLayer):
    plt.figure()
    plt.plot(range(1, maxCount/10+1), quaDict[str(i)], "r*-", sepDict[str(i)], "g+-", varDict[str(i)], "b.-")
    plt.savefig(cm.homeFld + "/Pictures/2013.7.9_"+str(i)+".jpg")
plt.show()
'''

#############################################################################
'''
siw = pd.W32
nLayer = 4
centroids = [64, 64, 32, 16]
dn = cm.init_destin(siw=siw, nLayer=nLayer, centroids=centroids)

ims = pd.ImageSouceImpl(32, 32)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.7.22/32")

sepDict = {}
varDict = {}
quaDict = {}
for i in range(nLayer):
    sepDict[str(i)] = []
    varDict[str(i)] = []
    quaDict[str(i)] = []

maxCount = 1600
for j in range(1, maxCount+1):
    if j % 10 == 0:
        print "Iteration " + str(j)
        for i in range(nLayer):
            sep = dn.getSep(i)
            var = dn.getVar(i)
            qua = dn.getQuality(i)
            sepDict[str(i)].append(sep)
            varDict[str(i)].append(var)
            quaDict[str(i)].append(qua)
    ims.findNextImage()
    f = ims.getGrayImageFloat()
    dn.doDestin(f)

cm.dcis(dn, 3)
cm.saveCens(dn, 3, cm.homeFld + "/Pictures/2013.7.25_layer3.jpg")
for i in range(nLayer):
    plt.figure()
    plt.plot(range(1, maxCount/10+1), quaDict[str(i)], "r*-", sepDict[str(i)], "g+-", varDict[str(i)], "b.-")
    plt.savefig(cm.homeFld + "/Pictures/2013.7.25_"+str(i)+".jpg")
plt.show()
'''

#############################################################################
'''
import charting as chart
import threading

siw = pd.W32
nLayer = 4
centroids = [64, 64, 32, 16]
dn = cm.init_destin(siw=siw, nLayer=nLayer, centroids=centroids)

ims = pd.ImageSouceImpl(32, 32)
cm.load_ims_fld(ims, cm.homeFld + "/Work/RECORD/2013.7.22/32")

class ChartingThread(threading.Thread):
    def run(self):
        chart.draw()
        
chart_thread = ChartingThread()

maxCount = 1600
for i in xrange(1, maxCount+1):
    if i % 10 == 0:
        print "Iteration " + str(i)
        j = nLayer-3
        var = dn.getVar(j)
        print "Variance: " +str(var)
        
        sep = dn.getSep(j)
        qua = dn.getQuality(j)
        
        chart.update([sep, var, qua])
        chart.draw()
    ims.findNextImage()
    f = ims.getGrayImageFloat()
    dn.doDestin(f)
'''

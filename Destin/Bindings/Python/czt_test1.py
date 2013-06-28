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
#cm.processFld("/home/teaera/Work/RECORD/2013.5.8/org", "/home/teaera/Work/RECORD/2013.5.8/pro_1")

#############################################################################
# 2013.5.7
'''
dn = cm.init_destin(extRatio=1)
ims = pd.ImageSouceImpl()
#cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_2")
#cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_2_2")
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013_5_8_1.jpg")
'''


#############################################################################
# 2013.5.9
'''
start_time = time.time()
dn = cm.init_destin(extRatio=2)
cm.train_2flds(dn, "/home/teaera/Work/RECORD/2013.5.8/pro_3", "/home/teaera/Work/RECORD/2013.5.8/pro_add_3", 10)
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
cm.cl.isNeedResize("/home/teaera/Work/RECORD/2013.5.8/pro_add_3/1.jpg")
f = cm.cl.get_float512()
cm.train_only(dn, f, 3000)
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.13
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl()
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=5000)
for i in range(8):
    #cm.saveCens(dn, i, "/home/teaera/Pictures/segmentation_face/%s.jpg"%(str(i)))
    cm.saveCens(dn, i, "/home/teaera/Pictures/original_face/%s.jpg"%(str(i)))
'''

#############################################################################
# 2013.5.15
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl()
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_only1")
cm.train_ims(dn, ims, maxCount=3500)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.15_fixed.jpg")
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.15_decay.jpg")
'''

#############################################################################
# 2013.5.15 centroids=[8,16,32,64,32,16,8,4]
'''
dn = cm.init_destin()
ims = pd.ImageSouceImpl()
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_2_2")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.15_decay_2.jpg")
'''

#############################################################################
# 2013.5.16
'''
dn = cm.init_destin(centroids=[8,16,32,64,64,64,32,16])
ims = pd.ImageSouceImpl()
#cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_4")
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.8/pro_5")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
#cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.16_1.jpg")
cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.16_2.jpg")
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
inFile = "/home/teaera/Downloads/orl_faces/s%s/1.pgm"
outFile = "/home/teaera/Work/EclipseWorkSpace/Test1_c1/images_c1/%s.pgm"
for i in range(40):
    os.system("cp " + inFile%(str(i+1)) + " " + outFile%(str(i+1)))
#
# STEP 2. Processing
cm.processFld("/home/teaera/Work/RECORD/2013.5.20/org_0", "/home/teaera/Work/RECORD/2013.5.20/pro_0")
'''
#------#
'''
>>>PART 2. About the generated results!
'''
'''
# Manually copied from out_1 to org_1...
# STEP 1. 
inFld = "/home/teaera/Work/RECORD/2013.5.20/org_1/"
files = os.listdir(inFld)
files.sort()
i = 1
for each in files:
    os.system("mv '" + inFld+each + "' " + inFld+str(i)+".jpg") # Because space, should use ''!!!
    i += 1
#
# STEP 2.
cm.processFld("/home/teaera/Work/RECORD/2013.5.20/org_1", "/home/teaera/Work/RECORD/2013.5.20/pro_1")
'''

#############################################################################
# 2013.5.20
'''
#dn = cm.init_destin(centroids=[12,18,24,30,30,24,18,12])
dn = cm.init_destin()
ims = pd.ImageSouceImpl()
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.20/pro_2/")
cm.train_ims(dn, ims, maxCount=5000)
cm.dcis(dn, 7)
'''

#############################################################################
# 2013.5.21
# STEP 1.
# Transform the downloaded images to gray-scale images first;
'''
cm.processFld("/home/teaera/Work/RECORD/2013.5.21/downloads", "/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine")
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
ims = pd.ImageSouceImpl()
cm.load_ims_fld(ims, "/home/teaera/Work/RECORD/2013.5.21/z_2")
cm.train_ims(dn, ims, maxCount=2000)
cm.dcis(dn, 7)
cm.saveCens(dn, 7, "/home/teaera/Pictures/2013.5.24_13.jpg")
'''

#############################################################################
# 2013.6.14
# I want to use matplotlib to draw the curve of validity!
# 2013.6.27
# Draw intra and inter seperately!
import matplotlib.pyplot as plt
#
centroids = [6,8,10,12,12,8,6,4]
size = 512*512
extRatio = 1
isUniform = True
dn = cm.init_destin(centroids=centroids, extRatio=extRatio)
ims = pd.ImageSouceImpl()
ims.addImage("/home/teaera/Work/RECORD/2013.5.8/pro_1/3.jpg")
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
    '''
    if i % 100 == 0:
        centroids[0] += 1
        dn.updateDestin_add(pd.W512, 8, centroids, isUniform,
                            size, extRatio, 0)
    '''
    ims.findNextImage()
    f = ims.getGrayImageFloat()
    #cl2.combineInfo_extRatio(f, size, extRatio, fOut)    
    dn.doDestin_c1(f)
#
'''
for i in range(8):
    plt.figure()
    plt.plot(range(len(valDict[str(i)])), valDict[str(i)], "r*-")
    plt.savefig("/home/teaera/Pictures/2013.6.18_"+str(i)+".jpg")
plt.show()
'''

for i in range(8):
    plt.figure()
    plt.plot(intraDict[str(i)], "r*-")
    plt.savefig("/home/teaera/Pictures/2013.6.28_intra_"+str(i)+".jpg")
    plt.figure()
    plt.plot(interDict[str(i)], "r*-")
    plt.savefig("/home/teaera/Pictures/2013.6.28_inter_"+str(i)+".jpg")
    plt.figure()
    plt.plot(valDict[str(i)], "r*-")
    plt.savefig("/home/teaera/Pictures/2013.6.28_validity_"+str(i)+".jpg")
'''
plt.show()
'''

'''
valMax = []
intraMax = []
interMax = []
for i in range(8):
    valDict[str(i)].sort(reverse=True)
    intraDict[str(i)].sort(reverse=True)
    interDict[str(i)].sort(reverse=True)
    #
    valMax.append(valDict[str(i)][0])
    intraMax.append(valDict[str(i)][0])
    interMax.append(valDict[str(i)][0])
'''
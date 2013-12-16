# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 21:14:05 2013

@author: ted
"""

import os, errno
import pydestin as pd
import cv2.cv as cv
import shutil
import czt_mod as czm

def train():
    for i in range(train_frames):
        if not ims.grab():
            continue        
        dn.doDestin(ims.getOutput())
        dn.printBeliefGraph(7,0,0)
    dn.save("minedtree.dst")
    return
            
def getTrees():
    dn.setIsPOSTraining(False) # freeze training
    for i in range(mine_frames):
        if not ims.grab():
            print "could not grab frame"
            continue;
            
        dn.doDestin(ims.getOutput())
        tm.addTree()
    tm.timeShiftTrees()
        
def displayMinedTree(index):
    if index >= found:
        print "out of bounds"
        return
    tm.displayMinedTree(index)
    cv.WaitKey(200)

def displayAllTrees():
    for i in range(found):
        print "Showing %i of %i" % (i,found - 1)
        tm.printMinedTreeStructure(i)
        tm.displayMinedTree(i)
        cv.WaitKey()
        

def displayCentroidImages(layer):
    dn.displayLayerCentroidImages(layer)
    cv.WaitKey(200)


# matches subtrees to their destin image
def matchSubtrees():
    for i in range(tm.getMinedTreeCount()):
        matches = tm.matchSubtree(i)
        print "found tree #%i matches:" % (i),
        for j in range(matches.size()):
            print str(matches[j]),
        print

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
def saveResults(run_id):
    out_dir="tree_mining_runs/"+run_id+"/"
    mkdir(out_dir)
    for l in range(layers):
        dn.saveLayerCentroidImages(l, out_dir+"layer_"+str(l)+".png")
        
    f = open(out_dir+"matches.txt",'w')
    
    for t in range(tm.getMinedTreeCount()):
        tm.saveMinedTreeImg(t, out_dir+"subtree_%i.png"%(t))
        matches = tm.matchSubtree(t)
        f.write("tree #%i matches input image: "%(t))
        for j in range(matches.size()):
            f.write(str(matches[j])+" ")
        f.write("\n")
        
    f.close()
    
    f = open(out_dir+"treepath.txt",'w')
    g = open(out_dir+"treepath_structure.txt",'w')
    for t in range(tm.getMinedTreeCount()):
        f.write("tree #%i: %s\n" % (t, tm.getMinedTreeAsString(t) ) )
        g.write("tree #%i:\n%s\n" % (t, tm.getMinedTreeStructureAsString(t) ) )
        
    f.close()
    g.close()
    
    shutil.copy(dst_save_file, out_dir+"network_save.bin")
    
    for i, l in enumerate(letters):
        shutil.copy("%s%s.png" % (img_path,l), out_dir+"input_img_%i.png"%(i))
        

## Init destin
centroids = [2, 2, 4, 8, 32, 16, 8, 3]
layers = len(centroids)
dn = pd.DestinNetworkAlt(pd.W512, layers, centroids, True)
dn.setBeliefTransform(pd.DST_BT_P_NORM)
#dn.setBeliefTransform(pd.DST_BT_NONE)
#dn.setBeliefTransform(pd.DST_BT_BOLTZ)

uniform_temp = 2
temperatures = []
for i in range(8):
    temperatures.append(uniform_temp)
#temperatures = [5, 5, 10, 20, 40, 20, 16, 6]

dn.setTemperatures(temperatures)
dn.setIsPOSTraining(True)

dn.setCentImgWeightExponent(4)

ims = pd.ImageSouceImpl(512, 512) 

letters = "+LO"
img_path = czm.homeFld + "/Pictures/treeminingletters/"
for l in letters:    
    ims.addImage("%s%s.png" % (img_path, l))
    
dst_save_file="+LO.dst"

## setup tree manager
bottom_layer = 4
tm = pd.DestinTreeManager(dn, bottom_layer)

train_frames = 1600
mine_frames = len(letters) + layers - 1

#train()
dn.load(dst_save_file)
getTrees()    

support = 2
found = tm.mine(support)
print "found %i trees" % (found)

matchSubtrees()
displayCentroidImages(7)    

saveResults("2")
print "click on tree images to continue"
displayAllTrees()



    
    
    

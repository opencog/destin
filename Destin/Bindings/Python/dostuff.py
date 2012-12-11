#!/usr/bin/python
import pydestin as pd

import opencv.highgui as cvhg

def array_to_pointer(arr):
    ia = pd.SWIG_IntArray(len(arr))
    for x in range(len(arr)):
        ia[x] = arr[x]
    return ia


centroids = [10,5,4,4,4,4,4,2]
             
#centroids = [2,2,2,2]
layers = len(centroids)
top_layer = layers - 1

layers_to_enum = {
        1: pd.W4,
        2: pd.W8,
        3: pd.W16,
        4: pd.W32,
        5: pd.W64,
        6: pd.W128,
        7: pd.W256,
        8: pd.W512}

img_width = layers_to_enum[layers]

print "iamge width is " + str(img_width)

dn = pd.DestinNetworkAlt(img_width, layers, centroids)

top_node = dn.getNode(top_layer, 0, 0)




vs = pd.VideoSource(False, "hand.AVI")
vs.setSize(img_width, img_width)

vs.enableDisplayWindow()

layerMask = pd.SWIG_UInt_Array_frompointer(dn.getNetwork().layerMask)

opened_windows = []

#make it so windows are more responsive
#cvhg.cvStartWindowThread()

#viz = pd.GenerativeVisualizer(dn.getNetwork())

def cls():
    import os
    os.system('clear')
    

def getNodeChild(parent_node, child_num):
    return pd.SWIG_Node_p_Array_getitem(parent_node.children, child_num)
                                          
def doFrame():
    if(vs.grab()):
        dn.doDestin(vs.getOutput())
        return True
    else:
        return False


def doFrames(frames):
    for i in range(frames):
        if not doFrame():
            return False
    return True



def dlf(layer, start, end):
    pd.DisplayLayerFeatures(dn.getNetwork(), layer, start, end)

def doFramesWithFeatures(frames, layer, start_node, end_node):
    for i in range(frames):
        if(vs.grab()):
            dn.doDestin(vs.getOutput())
            dlf(layer, start_node, end_node)
        else:
            return False
    return True

def doFramesWithWinningGrid(frames, layer):
    for i in range(frames):
        doFrame()
        dn.printWinningCentroidGrid(layer)

def doFramesWithCallback(frames, callback):
    for i in range(frames):
        doFrame()
        callback()

def beliefAndGridCallback():
    dn.printBeliefGraph(top_layer, 0, 0)
    dn.imageWinningCentroidGrid(1, 4)
    dlf(0, 0, 1)
    
def freezeTraining():
    dn.setIsPOSTraining(False)
    #dn.setIsPSSATraining(False)

def freezeLayer(layer):
    layerMask[layer] = 0

def unfreezeLayer(layer):
    layerMask[layer] = 1

def unfreezeAllLayers():
    for i in range(layers):
        layerMask[i] = 1



def freezeTopCentroidsExcept(lucky_centroid):
    if lucky_centroid == None:
        return
    starv = pd.SWIG_FloatArray_frompointer(top_node.starv)
    small = 1e-6
    for c in range(top_node.nb):
        starv[c] = 1

    starv[lucky_centroid] = small

def teachCentroid(centroid, frames=20):
    unfreezeLayer(top_layer)
    global lucky_centroid
    lucky_centroid = centroid
    go(frames)
    lucky_centroid = None
    freezeTraining()
    

def slowFreeze(start_layer, end_layer, frames_between):
    for l in range(start_layer, end_layer + 1):
        go(frames_between)
        freezeLayer(l)

def arrangeWindows():
    windows_wide = 4
    window_width = 256
    top_left = 0
    top_rigt = 0
    x = 0
    y = 0
    n = len(opened_windows)
    for i, w in enumerate(opened_windows):
        cvhg.cvResizeWindow(w, window_width, window_width)
        r, c = divmod(i, windows_wide)
        cvhg.cvMoveWindow(w, c * window_width, r * window_width)
    

lucky_centroid = None
def the_callback():
    zoom = 2
    global opened_windows
    opened_windows = []
    for i in range(8):
        wn = "Layer " + str(i)
        opened_windows.append(wn)
        dn.imageWinningCentroidGrid(i, zoom, wn)
        zoom*=2
        
    dn.printBeliefGraph(top_layer, 0, 0)
    freezeTopCentroidsExcept(lucky_centroid)
    dn.displayFeatures(1, 0, 1)
    
def go(frames=20):
    doFramesWithCallback(frames, the_callback)

def reportParentAndChildren(parent_layer, pr, pc):
    cls()
    L = parent_layer - 1
    dn.printNodeBeliefs(L, pr * 2,     pc * 2    )
    dn.printNodeBeliefs(L, pr * 2,     pc * 2 + 1)
    dn.printNodeBeliefs(L, pr * 2 + 1, pc * 2    )
    dn.printNodeBeliefs(L, pr * 2 + 1, pc * 2 + 1)
    
    go(1)
    dn.printNodeObservation(parent_layer, pr, pc)
  

#dn.load("hand.dst")
#freezeTraining()
#go(1)
#arrangeWindows()
#go(1)

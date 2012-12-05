#!/usr/bin/python
import pydestin as pd


def array_to_pointer(arr):
    ia = pd.SWIG_IntArray(len(arr))
    for x in range(len(arr)):
        ia[x] = arr[x]
    return ia



centroids = [2,3,3,3,4,4,4,2]
layers = len(centroids)
top_layer = layers - 1
dn = pd.DestinNetworkAlt(pd.W512, layers, centroids)

top_node = dn.getNode(top_layer, 0, 0)


vs = pd.VideoSource(False, "hand.MOV")
vs.enableDisplayWindow()

layerMask = pd.SWIG_UInt_Array_frompointer(dn.getNetwork().layerMask)

#viz = pd.GenerativeVisualizer(dn.getNetwork())

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
    lucky_centroid = centroid
    go(frames)
    lucky_centroid = None
    freezeTraining()
    

def slowFreeze(start_layer, end_layer, frames_between):
    for l in range(start_layer, end_layer + 1):
        go(frames_between)
        freezeLayer(l)

def the_callback():
    dn.imageWinningCentroidGrid(0, 4, "winning layer 0")
    dn.imageWinningCentroidGrid(1, 8, "winning layer 1")
    dn.imageWinningCentroidGrid(2, 16, "winning layer 2")
    #dn.imageWinningCentroidGrid(3, 16, "winning layer 3")
    #dn.imageWinningCentroidGrid(4, 16, "winning layer 4")
    dn.imageWinningCentroidGrid(5, 16, "winning layer 5")
    dn.printBeliefGraph(top_layer, 0, 0)
    freezeTopCentroidsExcept(lucky_centroid)
    
def go(frames=200):
    doFramesWithCallback(frames, the_callback)

        
lucky_centroid = None

    

#dn.load("hand.dst")
#freezeTraining()


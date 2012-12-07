#!/usr/bin/python
import pydestin as pd


def array_to_pointer(arr):
    ia = pd.SWIG_IntArray(len(arr))
    for x in range(len(arr)):
        ia[x] = arr[x]
    return ia


centroids = [3,3,3,3,4,4,4,2]
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

dn = pd.DestinNetworkAlt(img_width, layers, centroids)

top_node = dn.getNode(top_layer, 0, 0)


vs = pd.VideoSource(False, "hand.MOV")
vs.setSize(img_width, img_width)
vs.enableDisplayWindow()

layerMask = pd.SWIG_UInt_Array_frompointer(dn.getNetwork().layerMask)

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
    lucky_centroid = centroid
    go(frames)
    lucky_centroid = None
    freezeTraining()
    

def slowFreeze(start_layer, end_layer, frames_between):
    for l in range(start_layer, end_layer + 1):
        go(frames_between)
        freezeLayer(l)

def the_callback():
    zoom = 2
    for i in range(8):
        dn.imageWinningCentroidGrid(i, zoom, "Layer " + str(i))
        zoom*=2
        
    dn.printBeliefGraph(top_layer, 0, 0)
    freezeTopCentroidsExcept(lucky_centroid)
    
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
  
    
lucky_centroid = None



def printio(row, col):
    printio_n(dn.getNode(0,row,col))

def printio_n(node):
    io = pd.SWIG_UInt_Array_frompointer(node.inputOffsets)
    for i in range(4):
        for j in range(4):
            print io[i*4 + j],
        print
    
def printio_i(node_index):
    nodes = pd.SWIG_NodeArray_frompointer(dn.getNetwork().nodes)
    printio_n(nodes[node_index])
    

    

#dn.load("hand.dst")
#freezeTraining()


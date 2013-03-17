#!/usr/bin/python
import pydestin as pd
import time
import cv2.cv

cvhg = lambda: None
cvhg.cvResizeWindow = cv2.cv.ResizeWindow
cvhg.cvMoveWindow = cv2.cv.MoveWindow
cvhg.cvWaitKey = cv2.cv.WaitKey

centroids = [5,5,5,5,5,5,5,5]
            
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

dn = pd.DestinNetworkAlt(img_width, layers, centroids, True)

dn.setBeliefTransform(pd.DST_BT_P_NORM)

ct=2.0
dn.setTemperatures([ct,ct,ct,ct,ct,ct,ct,ct])
dn.setFixedLearnRate(.1)


top_node = dn.getNode(top_layer, 0, 0)

vs = pd.VideoSource(False, "hand.m4v")
vs.setSize(img_width, img_width)

vs.enableDisplayWindow()

layerMask = pd.SWIG_UInt_Array_frompointer(dn.getNetwork().layerMask)

opened_windows = []

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
    printStats()
    freezeTopCentroidsExcept(lucky_centroid)
    cent = the_callback.do_cent_image_num
    if the_callback.do_cent_image:
        dn.displayCentroidImage(the_callback.dfl, cent, the_callback.cent_image_disp_width )

the_callback.do_cent_image = False
the_callback.do_cent_image_num = 0
the_callback.dfl = 7
the_callback.cent_image_disp_width = 512

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
  
def printCentImage(layer, cent):
    l = pd.GetCentroidImageWidth(dn.getNetwork(), layer)
    l = l * l
    fa = pd.SWIG_FloatArray_frompointer( dn.getCentroidImage(layer, cent))
    for i in range(l):
        print fa[i]

        
#display centroid image       
def dci(layer, cent, equalize_hist = False, exp_weight = 4):
    dn.setCentImgWeightExponent(exp_weight)
    dn.displayCentroidImage(layer, cent, 512, equalize_hist)
    cv2.waitKey(100)

def wk(time=100):
    cv2.waitKey(time)
    
def printStats():
    n = dn.getNode(7,0,0)
    print "Winner: %i" %(n.winner)
    starv =  pd.SWIG_FloatArray_frompointer( n.starv )
    for c in range(n.nb):
        print "starv %i: %f" % (c, starv[c])
        
    print ""
    
def incrementTrain():
    for l in range(layers):
        freezeTraining()        
        unfreezeLayer(l)
        go(200)
        
def eatDogFood(centroid):
    if centroid >= dn.getBeliefsPerNode(top_layer):
        print "out of bounds"
        return
    dn.displayCentroidImage(top_layer, centroid)
    dn.setCentImgWeightExponent(8)
    img = dn.getCentroidImage(top_layer, centroid)
    freezeTraining()
    for i in range(layers):
        dn.doDestin(img)
    the_callback()
    cv2.waitKey(300)
   
def cycleCentroidImages(layer):
    for c in range(dn.getBeliefsPerNode(layer)):
        dn.displayCentroidImage(layer, c, 256, True)
        cv2.waitKey(300)               
        time.sleep(1)

dn.setIsPOSTraining(True)
dn.setCentImgWeightExponent(4)

go(5)
arrangeWindows()
go(500)



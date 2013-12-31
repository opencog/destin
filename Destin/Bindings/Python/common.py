# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:46:36 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv
import time
import czt_mod as czm
import os, errno

layers_to_enum = {
        1: pd.W4,
        2: pd.W8,
        3: pd.W16,
        4: pd.W32,
        5: pd.W64,
        6: pd.W128,
        7: pd.W256,
        8: pd.W512}


network = None
layers = 0
top_layer = 0
video_source = None

"""layerMask
Array of integers of size of the number of layers
Array element value of 0 means the layer is not training.
Array element value of 1 means the layer is training.
"""
layerMask = None

""" after a frame is shown this is called """
def the_callback():
    network.printBeliefGraph(top_layer, 0, 0)

def top_node():
    # function instead of saving the reference in case
    # the node is destroyed during reloadingd
    return network.getNode(top_layer, 0, 0)
    
def init(img_width, centroids=[2,4,16,32,64,32,16,8], 
         video_file=czm.homeFld + "/Dropbox/destin/moving_square.mov", 
         temperature=2.0,
         learn_rate=0.1,
         layer_widths=None):
    global network, layers, video_source, layerMask, top_layer
    layers = len(centroids)
    top_layer = layers - 1
    network = pd.DestinNetworkAlt(img_width, layers, centroids, True, layer_widths)

    temps = []
    for i in xrange(layers):        
        temps.append(temperature)    
    
    pd.SetLearningStrat(network.getNetwork(), pd.CLS_FIXED)
    
    network.setTemperatures(temps)
    network.setFixedLearnRate(learn_rate)
    network.setBeliefTransform(pd.DST_BT_P_NORM)

    layerMask = pd.SWIG_UInt_Array_frompointer(network.getNetwork().layerMask)

    video_source = pd.VideoSource(False,video_file)
    video_source.setSize(img_width, img_width)


"""
Clear console
"""
def cls():
    import os
    os.system('clear')
    
def getNodeChild(parent_node, child_num):
    return pd.SWIG_Node_p_Array_getitem(parent_node.children, child_num)
                                          
def doFrame():
    if(video_source.grab()):
        network.doDestin(video_source.getOutput())
        return True
    else:
        return False


def doFrames(frames):
    for i in xrange(frames):
        if not doFrame():
            return False
    return True

def doFramesWithWinningGrid(frames, layer):
    for i in xrange(frames):
        doFrame()
        network.printWinningCentroidGrid(layer)

def doFramesWithCallback(frames, callback):
    for i in xrange(frames):
        doFrame()
        callback(i)

def beliefAndGridCallback():
    network.printBeliefGraph(top_layer, 0, 0)
    network.imageWinningCentroidGrid(1, 4)
    
def freezeTraining():
    network.setIsPOSTraining(False)
    #dn.setIsPSSATraining(False)

def freezeLayer(layer):
    layerMask[layer] = 0

def unfreezeLayer(layer):
    layerMask[layer] = 1

def unfreezeAllLayers():
    for i in xrange(layers):
        layerMask[i] = 1

def freezeTopCentroidsExcept(lucky_centroid):
    if lucky_centroid == None:
        return
    starv = pd.SWIG_FloatArray_frompointer(top_node().starv)
    small = 1e-6
    for c in xrange(top_node().nb):
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
    for l in xrange(start_layer, end_layer + 1):
        go(frames_between)
        freezeLayer(l)

def go(frames=20):
    doFramesWithCallback(frames, the_callback)
  
def printCentImage(layer, cent):
    l = pd.GetCentroidImageWidth(network.getNetwork(), layer)
    l = l * l
    channel = 0
    fa = pd.SWIG_FloatArray_frompointer(network.getCentroidImage(channel, layer, cent))
    for i in xrange(l):
        print fa[i]

def dci(layer, cent, equalize_hist = False, exp_weight = 4):
    """ Display centroid image """
    num_centroids = network.getBeliefsPerNode(layer)
    if(cent >= num_centroids):
        print "centoid out of bounds."
        return
    network.setCentImgWeightExponent(exp_weight)
    network.displayCentroidImage(layer, cent, 512, equalize_hist)
    wk()
    
def dcis(layer=top_layer, exp_weight = 4):
    """ Display centroid images.
    All centroid images in a layer are shown at once.
    """
    network.setCentImgWeightExponent(exp_weight)
    network.displayLayerCentroidImages(layer,600)
    wk()

def wk(time=100):
    """ Calls opencv waitkey, so that opencv images can be drawn."""
    cv2.waitKey(time)
    
def printStats():
    """ Prints the winning centriod index and starvation trace for the top node."""
    n = top_node()
    print "Winner: %i" %(n.winner)
    starv =  pd.SWIG_FloatArray_frompointer( n.starv )
    for c in xrange(n.nb):
        print "starv %i: %f" % (c, starv[c])
        
    print ""


def printFPS():
    t = time.time()
    print "FPS:", round(1.0 / (t - printFPS.lastTime))
    printFPS.lastTime = t
printFPS.lastTime=0
    
    
def incrementTrain():
    for l in xrange(layers):
        freezeTraining()        
        unfreezeLayer(l)
        go(200)
    

def eatOwnDogFood(centroid):
    """Feed the top node centroid image back into the network.
    Hopefully the network will then have its believe distribution
    spike on that centroid.
    """
    if centroid >= network.getBeliefsPerNode(top_layer):
        print "out of bounds"
        return
    network.displayCentroidImage(top_layer, centroid)
    network.setCentImgWeightExponent(8)
    channel = 0
    #TODO: update to work with multi channels
    img = network.getCentroidImage(channel, top_layer, centroid)
    freezeTraining()
    for i in xrange(layers):
        network.doDestin(img)
    the_callback()
    wk()
   
def cycleCentroidImages(layer):
    """ Cycles through all the centroid images for a layer
    Next centroid image is shown each time the centroid display is clicked.
    """
    for c in xrange(network.getBeliefsPerNode(layer)):
        network.displayCentroidImage(layer, c, 256, True)
        wk()
        time.sleep(1)
    
    
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def saveCentroidLayerImages(network, root_dir, run_id, image_width, weight_exponent):
    run_dir = root_dir + "/"+run_id+"/"
    mkdir(run_dir)
    border_width = 5

    network.setCentImgWeightExponent(weight_exponent)
    network.updateCentroidImages()
        
    sort_orders = []
    for i in xrange(network.getLayerCount()):
        file_name = run_dir+"layer_%i_images_exp_%s.jpg" % (i, "{0:.2f}".format(weight_exponent))
        sort_orders.append(network.sortLayerCentroids(i))
        network.saveLayerCentroidImages(i, file_name, image_width, border_width, sort_orders[i])
        
    network.setCentImgWeightExponent(1)
    network.updateCentroidImages()
    for i in xrange(network.getLayerCount()):
        network.saveLayerCentroidImages(i, run_dir+"layer_%i_images_exp_1.jpg" % (i), image_width, border_width, sort_orders[i])        
        

def saveCenImages(network, root_dir, run_id, layer, centroid_image_width, weight_exponent):
    run_dir = root_dir + "/"+run_id+"/"
    mkdir(run_dir)

    # with centroid weight = 1 
    orig_dir = run_dir+"orig/"
    
    # with centroid weight = 1 and enhanced = true
    orige_dir = run_dir+"orig_e/"
    
    # store centroid with higher weighting
    highweighted_dir = run_dir+"highweight/"
    
    highweightede_dir = run_dir+"highweight_e/"

    for d in [orig_dir, orige_dir, highweighted_dir, highweightede_dir]:
        mkdir(d)
    
    # Save centriod images
    bn = network.getBeliefsPerNode(layer)
        
    network.setCentImgWeightExponent(1)
    network.updateCentroidImages()
    
    for i in range(bn):
        f = "%02d_%02d_%s.png" % (layer, i,run_id)
        # save original
        fn = orig_dir + f 
        network.saveCentroidImage(layer, i, fn, centroid_image_width, False )
    
        # save original enhanced
        fn = orige_dir + f
        network.saveCentroidImage(layer, i, fn, centroid_image_width, True )
    
    
    network.setCentImgWeightExponent(weight_exponent)
    network.updateCentroidImages()
    for i in range(bn):
        f = "%02d_%02d_%s.png" % (layer, i,run_id)
        fn = highweighted_dir + f 
        network.saveCentroidImage(layer, i, fn, centroid_image_width, False )
        fn = highweightede_dir + f
        network.saveCentroidImage(layer, i, fn, centroid_image_width, True )
        
def displayAllLayers(network, image_weight_exponent, sortCentroids=True):
    """
        diplay centriod layer images
        Left key, go to previous layer
        Esc to cancel
        Any other key to move to next image
    """
    esc_key = 1048603
    left_arrow = 1113937
    current_image = 0
    close_button = -1
    network.setCentImgWeightExponent(image_weight_exponent)
    network.updateCentroidImages()
    while True:
        if sortCentroids:
            sort_order = network.sortLayerCentroids(current_image)
            network.displayLayerCentroidImages(current_image, 1000, 5, "Centroid Images", sort_order)
        else:
            network.displayLayerCentroidImages(current_image, 1000, 5, "Centroid Images")
            
        pressed = cv.WaitKey()
        if pressed == esc_key or pressed == close_button:
            return
        elif pressed == left_arrow:
            current_image = current_image - 1
        else:
            current_image = current_image + 1
        if current_image < 0:
            current_image = network.getLayerCount() - 1
        if current_image >= network.getLayerCount():
            current_image = 0
            
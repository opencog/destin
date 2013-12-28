# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:52:28 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv
import charting as chart
import common as cm

"""
This script defines a "go()" function which will train DeSTIN on CIFAR images ( see http://www.cs.toronto.edu/~kriz/cifar.html )
and then presents the DeSTIN beliefs on a self organizing map ( SOM ).

See http://www.mediafire.com/view/?17ehjc28z922g#um21dwtl1lz1f8v for a screen shot of this script.
Each colored dot represents an image. The color of the dot is determined by the image class such as dog or airplane.
The self organizing map should put simular images near each other.

If you click on SOM it will show you CIFAR image of the nearest dot.

"""

# Loads the destin config from  cifar_experiment_config.py
from cifar_experiment_config import *

dn = pd.DestinNetworkAlt(pd.W32, layers, centroids, True, None, image_mode)

# I turned off using previous beliefs in DeSTIN because I dont
# think they would be useful in evaluating static images.
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

# BeliefExporter - picks which beliefs from destin to show to the SOM
be = pd.BeliefExporter(dn, bottom_belief_layer)

def getCifarFloatImage():
    if image_mode == pd.DST_IMG_MODE_RGB:
        return cs.getRGBImageFloat()
    elif image_mode == pd.DST_IMG_MODE_GRAYSCALE:
        return cs.getGrayImageFloat()
    else:
        raise Exception("unsupported image mode")

def train_stages(iterations_per_layer_list):
    """
    Train each layer one at a time, using the 
    list of iterations per layer.
    
    If it's not  a list then just train all layers at once.
    """
    if type(iterations_per_layer_list) == list:        
        for layer, iterations in enumerate(iterations_per_layer_list):
            train_destin(iterations, layer)
    else:
        train_destin(iterations_per_layer_list)
        

#train the network
def train_destin(training_iterations, train_only_layer=-1):
    global image_ids
    image_ids = []
    
    # only enable one layer if specified
    if train_only_layer != -1:
        for i in xrange(layers):
            dn.setLayerIsTraining(i, False)
        dn.setLayerIsTraining(train_only_layer, True)
        
    for i in range(training_iterations):
        if i % 100 == 0:
            print "Training DeSTIN iteration: " + str(i)
            
            #chart.update([dn.getQuality(top_layer), dn.getVar(top_layer), dn.getSep(top_layer)])
            #chart.update([dn.getVar(top_layer), dn.getSep(top_layer)])
            #report_layer = 0
            #variance = dn.getVar(report_layer)
            #seperation = dn.getSep(report_layer)
            #quality = dn.getQuality(report_layer)
            #qual_moving_average = moving_average(quality)
            #chart.update([variance, seperation])
            chart.update(dn.getLayersQualities())
            if i%200 == 0:
                chart.draw()
           # print "Qual: %f, Variance: %f, seperation: %f, average: %f" % (quality, variance, seperation, qual_moving_average)
        
        #find an image of an enabled class
        cs.findNextImage()

        #save the image's id / index for layer replay
        image_ids.append(cs.getImageIndex())

        dn.doDestin(getCifarFloatImage())
    
    dn.save( "saved.dst")


def showDestinImage(i):
    im_id = i % len(image_ids)
    cs.setCurrentImage(image_ids[im_id])
    dn.clearBeliefs()
    for j in range(layers):
        
        dn.doDestin(getCifarFloatImage())
            
#Show the cifar images, and write the beliefs to the mat file.
def dump_beliefs():
    #turn off destin training so
    #its beliefs for a given image stay fixed
    for j in range(layers):
        dn.setLayerIsTraining(j, False)
        
    for i in range(supervise_train_iterations):
        # show DeSTIN a CIFAR image
        showDestinImage(i)
        
        # write the cifar image type/class ( i.e. cat / dog )
        # and the current beliefs to the mat file
        # print "class label: %i " % (cs.getImageClassLabel())
        be.writeBeliefToDisk(cs.getImageClassLabel())
        if i % 100 == 0:
            print "Iteration %d" % i
    
def showCifarImage(id):
     cs.setCurrentImage(id)
     ci = cs.getColorImageMat()
     pd.imshow("Cifar Image: " + str(id), ci)
     cv.WaitKey(500)

def go():
    train_stages(training_iterations)
    print "Training Supervision..."
    # show cifar images, and dump resulting beliefs to a .txt file
    #dump_beliefs()
    be.closeBeliefFile()
    print "Done."
    
def dcis(layer = 0):
    dn.displayLayerCentroidImages(layer, 1000)
    cv.WaitKey(100)


#Start it all up
go()

cm.saveCentroidLayerImages(dn, experiment_save_dir, run_id, save_image_width, weight_exponent)
chart.savefig("%s/%s/chart_%s.jpg" % ( experiment_save_dir, run_id, run_id))
cm.displayAllLayers(dn, weight_exponent)
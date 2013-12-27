# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:52:28 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv
import charting as chart


"""
This script defines a "go()" function which will train DeSTIN on CIFAR images ( see http://www.cs.toronto.edu/~kriz/cifar.html )
and then presents the DeSTIN beliefs on a self organizing map ( SOM ).

See http://www.mediafire.com/view/?17ehjc28z922g#um21dwtl1lz1f8v for a screen shot of this script.
Each colored dot represents an image. The color of the dot is determined by the image class such as dog or airplane.
The self organizing map should put simular images near each other.

If you click on SOM it will show you CIFAR image of the nearest dot.

"""

from cifar_experiment_config import *

def getCifarFloatImage():
    if image_mode == pd.DST_IMG_MODE_RGB:
        return cs.getRGBImageFloat()
    elif image_mode == pd.DST_IMG_MODE_GRAYSCALE:
        return cs.getGrayImageFloat()
    else:
        raise Exception("unsupported image mode")

#train the network
def train_destin():
    global image_ids
    image_ids = []
    for i in range(training_iterations):
        if i % 100 == 0:
            print "Training DeSTIN iteration: " + str(i), 
            
            #chart.update([dn.getQuality(top_layer), dn.getVar(top_layer), dn.getSep(top_layer)])
            #chart.update([dn.getVar(top_layer), dn.getSep(top_layer)])
            report_layer = 0
            variance = dn.getVar(report_layer)
            seperation = dn.getSep(report_layer)
            quality = dn.getQuality(report_layer)
            qual_moving_average = moving_average(quality)
            #chart.update([variance, seperation])
            chart.update([quality, qual_moving_average])
            if i%200 == 0:
                chart.draw()
            print "Qual: %f, Variance: %f, seperation: %f, average: %f" % (quality, variance, seperation, qual_moving_average)

        #find an image of an enabled class
        cs.findNextImage()

        #save the image's id / index for layer replay
        image_ids.append(cs.getImageIndex())

        #clear beliefs so previous images dont affect this one
        dn.clearBeliefs()

        #disable all training, then re-enable the layers
        #one by one while the image "signal" propagates up the
        #heirarchy over the iterations
        for j in range(layers):
            dn.setLayerIsTraining(j, False)
        for j in range(layers):
            dn.setLayerIsTraining(j, True)
            dn.doDestin(getCifarFloatImage())
            
        #let it train for 2 more times with all layers training
        for j in range(2):
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
    train_destin()
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
dcis(0)

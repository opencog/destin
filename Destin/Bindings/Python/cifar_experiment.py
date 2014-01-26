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
This script defines a "go()" function which will train DeSTIN on CIFAR images 
( see http://www.cs.toronto.edu/~kriz/cifar.html )
"""

# Loads the destin config from  cifar_experiment_config.py
from cifar_experiment_config import *

cs = pd.CifarSource(cifar_dir, cifar_batches)
cs_test = pd.CifarSource(cifar_dir, cifar_test_batch)
top_layer = len(centroids) - 1

for the_cs in [cs, cs_test]:
    the_cs.disableAllClasses()
    for cifar_class in cifar_classes_enabled:
        the_cs.setClassIsEnabled(cifar_class, True)

dn = pd.DestinNetworkAlt(pd.W32, layers, centroids, True, None, image_mode)

# I turned off using previous beliefs in DeSTIN because I dont
# think they would be useful in evaluating static images.
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

# BeliefExporter - picks which beliefs from destin to show to the SOM
be = pd.BeliefExporter(dn, bottom_belief_layer)

def getCifarFloatImage(cifar_source):
    if image_mode == pd.DST_IMG_MODE_RGB:
        return cifar_source.getRGBImageFloat()
    elif image_mode == pd.DST_IMG_MODE_GRAYSCALE:
        return cifar_source.getGrayImageFloat()
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
def train_destin(destin_train_iterations, train_only_layer=-1):
  
    # only enable one layer if specified
    if train_only_layer != -1:
        for i in xrange(layers):
            dn.setLayerIsTraining(i, False)
        dn.setLayerIsTraining(train_only_layer, True)
        
    for i in range(destin_train_iterations):
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

        dn.doDestin(getCifarFloatImage(cs))
    
    dn.save( experiment_save_dir+"/network_"+run_id+".dst")


def dump_beliefs(output_filename, purpose, image_count, cifar_source):
    print "Dumping %s beliefs for %d images..." %(purpose, image_count)
    #turn off destin training so
    #its beliefs for a given image stay fixed
    dn.isTraining(False)
    cifar_source.setCurrentImage(-1)
    for i in xrange(image_count):
        cifar_source.findNextImage()
        dn.clearBeliefs()
        for j in xrange(layers): # let the image propagate through all layers
            dn.doDestin(getCifarFloatImage(cifar_source))
            
        # write the cifar image type/class ( i.e. cat / dog )
        # and the current beliefs to the mat file
        be.writeBeliefToDisk(cifar_source.getImageClassLabel(), output_filename)
        if i % 100 == 0:
            print "Image %d of %d" % (i, image_count)
    be.closeBeliefFile()
        
#Show the cifar images, and write the beliefs to the mat file.
def dump_training_beliefs():
    dump_beliefs(output_training_beliefs_filename, "training", n_output_training_features, cs)
    
def dump_testing_beliefs():
    dump_beliefs(output_test_beliefs_filename, "testing", n_output_testing_features, cs_test)
     
def displayCifarImage(id):
     cs.setCurrentImage(id)
     ci = cs.getColorImageMat()
     pd.imshow("Cifar Image: " + str(id), ci)
     cv.WaitKey(500)

def dcis(layer = 0):
    """ display centroid images """
    dn.displayLayerCentroidImages(layer, 1000)
    cv.WaitKey(100)
    
def go():
    train_stages(destin_train_iterations)

    cm.saveCentroidLayerImages(dn, experiment_save_dir, run_id, save_image_width, weight_exponent)
    chart.savefig("%s/%s/chart_%s.jpg" % ( experiment_save_dir, run_id, run_id))
    
    # show training cifar images and dump resulting beliefs to a .txt file 
    # to be used by supervising algorithm ( i.e. neural net)    
    dump_training_beliefs()

    # show testing cifar images and dump resulting beliefs to a .txt file 
    # to be used fpr test the supervising algorithm.
    dump_testing_beliefs()
        
    print "Displaying centroid images: ..."
    cm.displayAllLayers(dn, weight_exponent)
    
#Start it all up
go()
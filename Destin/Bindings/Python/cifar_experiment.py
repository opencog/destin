# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 20:52:28 2013

@author: ted
"""

import pydestin as pd
import cv2.cv as cv
import charting as chart
import SimpleMovingAverage as sma

"""
This script defines a "go()" function which will train DeSTIN on CIFAR images ( see http://www.cs.toronto.edu/~kriz/cifar.html )
and then presents the DeSTIN beliefs on a self organizing map ( SOM ).

See http://www.mediafire.com/view/?17ehjc28z922g#um21dwtl1lz1f8v for a screen shot of this script.
Each colored dot represents an image. The color of the dot is determined by the image class such as dog or airplane.
The self organizing map should put simular images near each other.

If you click on SOM it will show you CIFAR image of the nearest dot.

"""

# Downlaod the required data at http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# Set this variable to the folder containing data_batch_1.bin to data_batch_5.bin
# cifar_dir = os.getenv("HOME") + "/Downloads/cifar-10-batches-bin"
cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"

cifar_batch = 1 #which CIFAR batch to use from 1 to 5
cs = pd.CifarSource(cifar_dir, cifar_batch)

#must have 4 layers because the cifar data is 32x32
layers = 4
centroids = [128,32,23,2]

image_mode = pd.DST_IMG_MODE_RGB
#image_mode = pd.DST_IMG_MODE_GRAYSCALE

# How many CIFAR images to train destin with. If larger than
# If this this is larger than the number of possible CIFAR images then some
# images will be repeated
training_iterations = 20000

supervise_train_iterations = 10000

is_uniform = True # uniform DeSTIN or not
dn = pd.DestinNetworkAlt(pd.W32, layers, centroids, True, None, image_mode)

# I turned off using previous beliefs in DeSTIN because I dont
# think they would be useful in evaluating static images.
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

# The som trains on concatenated beliefs starting from this layer to the top layer.
# If  bottom_belief_layer = 0 then it will use all the beliefs from all the layers.
# If bottom_belief_layer = 3 then only the top layer's beliefs will be used.
bottom_belief_layer = 2 

# BeliefExporter - picks which beliefs from destin to show to the SOM
be = pd.BeliefExporter(dn, bottom_belief_layer)

# How many times  at once an individual CIAR image should be shown to destin in one training iteration
# Should be at least 4 because it takes a few iterations for an image to propagate through all the layers.
iterations_per_image = 8

# This block picks which image classes to use.
# See http://www.cs.toronto.edu/~kriz/cifar.html for the possible image classes.
cs.disableAllClasses()
cs.setClassIsEnabled(0, True) #airplane
#cs.setClassIsEnabled(1, True) #automobile
#cs.setClassIsEnabled(2, True) #bird
#cs.setClassIsEnabled(3, True) #cat
cs.setClassIsEnabled(4, True) #deer
#cs.setClassIsEnabled(5, True) #dog
#cs.setClassIsEnabled(6, True) #frog
#cs.setClassIsEnabled(7, True) #horse
#cs.setClassIsEnabled(8, True) #ship
#cs.setClassIsEnabled(9, True) #truck

# which ids of the CIFAR images that were used in training
image_ids = []
top_layer = len(centroids) - 1

"""
consider belief propagation delay. For a static image, if there are
4 layers, then it will take 4 iterations for the upper layers to
start seeing anything, unless we implement a way to only enable 1
layer processing at a time to avoid the propagation delay.

Since the image is static, I dont think it would need previous belief recurrence.
I don't understand how the parent nodes would be able to help the bottom nodes.

If doing 8 iteration per image, then since half the time the top layers will be seeing
junk beliefs, the training on those top ones should be disabled
"""

# moving average for centroid quality graph
moving_average_period = 5
moving_average = sma.SimpleMovingAverage(moving_average_period)

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
    dn.displayLayerCentroidImages(layer)
    cv.WaitKey(100)
#Start it all up
go()
dcis(0)

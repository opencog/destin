# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:30:49 2013

@author: ted
"""
import pydestin as pd
import SimpleMovingAverage as sma


# Downlaod the required data at http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# Set this variable to the folder containing data_batch_1.bin to data_batch_5.bin
# cifar_dir = os.getenv("HOME") + "/Downloads/cifar-10-batches-bin"
cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"
experiment_save_dir="./cifar_experiment_runs"

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

save_image_width = 1000
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


weight_exponent = 4

################## Experiment Run Log ###################

"""
Run 2
It appears the quality doesn't change much after 5000 iterations.
Will try just 5000 to see if the images look the same.

Results: 
    Didn't change much as expected.
"""
training_iterations = 5000


"""
Run 3
"""
run_id = "003"
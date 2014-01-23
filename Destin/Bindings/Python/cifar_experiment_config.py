# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:30:49 2013

@author: ted
"""
import pydestin as pd
import SimpleMovingAverage as sma

"""
Important:
    Check this whole file for values of the variables.
    Each experiment run may override, the last set value
    is the one that matters.
"""

# Downlaod the required data at http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# Set this variable to the folder containing data_batch_1.bin to data_batch_5.bin
# cifar_dir = os.getenv("HOME") + "/Downloads/cifar-10-batches-bin"
cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"
experiment_save_dir="./cifar_experiment_runs"

cifar_batch = 1 #which CIFAR batch to use from 1 to 5
cifar_test_batch = 2 #which CIFAR batch to test trained destin with

output_training_beliefs_filename = "OutputTrainBeliefs.txt"
output_test_beliefs_filename = "OutputTestBeliefs.txt"

# This block picks which image classes to use.
# See http://www.cs.toronto.edu/~kriz/cifar.html for the possible image classes.
#0 airplane
#1 automobile
#2 bird
#3 cat
#4 deer
#5 dog
#6 frog
#7 horse
#8 ship
#9 truck
cifar_classes_enabled = [0,4]

#must have 4 layers because the cifar data is 32x32
layers = 4
centroids = [128,32,23,2]

image_mode = pd.DST_IMG_MODE_RGB
#image_mode = pd.DST_IMG_MODE_GRAYSCALE

# How many CIFAR images to train destin with. If larger than
# If this this is larger than the number of possible CIFAR images (10000) then some
# images will be repeated
training_iterations = 20000

# How many belief rows( features ) will be dumped to a beliefs file
# so that it can be used for supervised training ( for a neural network, or other classifier)
supervise_train_iterations = 10000

is_uniform = True # uniform DeSTIN or not

# The som trains on concatenated beliefs starting from this layer to the top layer.
# If  bottom_belief_layer = 0 then it will use all the beliefs from all the layers.
# If bottom_belief_layer = 3 then only the top layer's beliefs will be used.
bottom_belief_layer = 2 



# How many times  at once an individual CIAR image should be shown to destin in one training iteration
# Should be at least 4 because it takes a few iterations for an image to propagate through all the layers.
iterations_per_image = 8



# which ids of the CIFAR images that were used in training
image_ids = []

save_image_width = 1000 # width of centroid layer images saved to file.

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

# See weightParameter of Cig_CreateCentroidImages in cent_image_gen.c
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
Run 4
Undo the strategy of clearing beliefs between each one. 
I think for training then,centroids will be able to move easier.
Results:
    It seems to result in more colors on the bottom layers
"""
run_id = "004"
training_iterations = 10000

"""
Run 5

Try to train each layer in stages, starting from the bottom.

Results;
The quality seems to stabalize pretty quickly.

"""
run_id = "005"
training_iterations = [3000,3000,3000,3000]


"""
Run 6
See if adding more centroids on layer 1 will result in more diverse centroids

Results:
    It takes a little longer for layer 1 quality to stabalize
"""
run_id = "006"
centroids = [128,64,32,2]

"""
Run 7
See if adding more centroids to the top layer will result in more diverse
centroid images.

Resutls:
    The top centroid images were more unique. Before there was variance in the
    quality measure of the top layer when there was only 2 centroids, now
    the quality measure seemed to perfectly stabalize. It's hard to tell
    what the quality is. 
    
    Need to add ability to log the quality measures.
    
"""
run_id = "007"
centroids = [128,64,32,16]

"""
Run 8
Will try to determine if there's a practical limit to the top layer
centroids. Will they continue to be diverse as the number increases?

Results:
    A few centroids look pretty simulator, but they still look pretty diverse.
    It takes a little bit longer still for the quality of the top layer to 
    stabalize.
"""
run_id = "008"
centroids = [128,64,32,32]


"""
Run 9
Determine if doubling layer 1 to 128 centroid will maintain or increase diversity.
Determine if it resulted in more diversity in layer 2.

Results:
    Didn't decrease, but was harder to tell if it increased it. 
    Could probably write a was to sort the centroid images based on simularity.
    
Waffles gave pretty good results.
99.36% accuracy ( actually this was overtraining, cross validation reveals more like 79% accuracy)
"""
run_id="009"
centroids = [128,128,32,32]


"""
Run 10

Enable all cifar classes, see how the nn can predict the results

Results:
        Note: waffles reports Mean squared error, but it really means 1 - Accuracy rate when predicting discreet labels instead of continuous values
        
        run_10_01:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.005 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002    
            Accuracy: .4237
            Baseline Accuracy: .1069
        run_10_02:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 400 -learningrate 0.005 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002
            Accuracy: .4422
        run_10_03:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.002 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002
            file: nn3.model
            train time: 22m3.478s
            Mean squared error: 0.0348
            
        run_10_04:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.002
            file: nn4.model
            traintime: 8m34.904s
            Mean squared error: 0.098
                       
        run_10_05:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 100 -learningrate 0.005 -momentum 0.3 -windowepochs 100 -minwindowimprovement 0.002
            file: nn5.model
            traintime: 1m46.884s
            Mean squared error: 0.4967
            
        run_10_06:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.001 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.002
            file: nn6.model
            traintime: 8m2.960s
            Mean squared error: 0.1171
        
        run_10_07:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 100 -learningrate 0.002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.002 > nn7.model
            train time: 6m9.983s
            Mean squared error: 0.1126
            
        run_10_08:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.002 -momentum 0.1 -windowepochs 100 -minwindowimprovement 0.002 > nn8.model
            train time: 8m51.823s
            Mean squared error: 0.1063
            
        run_10_09:
            waffles_learn train -seed 0 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 200 -learningrate 0.002 -momentum 0.2 -windowepochs 100 -minwindowimprovement 0.001 > nn9.model
            train time: 8m43.846s
            Mean squared error: 0.098
            
Note: those previous runs were overfitting the data with too many neurons. 
The best results with a 3 fold crossvalidation came from:
    waffles_learn crossvalidate -seed 0 -reps 1 -folds 3 OutputBeliefs.txt.arff -labels 0 neuralnet -addlayer 160 -learningrate 0.002 -momentum 0.3 -windowepochs 1600 -minwindowimprovement .002
    Error rate: 0.2023 = 79.77% accuracy
            
"""

run_id="010"
cifar_classes_enabled = [0,1,2,3,4,5,6,7,8,9]


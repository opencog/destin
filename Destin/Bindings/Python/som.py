import time
import pydestin as pd

import opencv as cv
import opencv.highgui as hg

cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"
cs = pd.CifarSource(cifar_dir, 1)

#must be size 4 because the cifar data is 32x32
centroids = [10,5,5,5]
layers = 4

training_iterations = 5000

som_train_iterations = 10000

dn = pd.DestinNetworkAlt(pd.W32, layers, centroids)
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

iterations_per_image = 8

marker_width = 5

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


cs.disableAllClasses()
#cs.setClassIsEnabledByName("airplane", True)
#cs.setClassIsEnabledByName("cat", True)
#cs.setClassIsEnabledByName("ship", True)
cs.setClassIsEnabled(0, True)
cs.setClassIsEnabled(1, True)
cs.setClassIsEnabled(2, True)
cs.setClassIsEnabled(3, True)



batch = 0

image_ids = []

be = pd.BeliefExporter(dn, 2)

som = None

#train the network
def train_destin():
    global image_ids
    
    for i in range(training_iterations):
        if i % 50 == 0:
            print "Training DeSTIN iteration: " + str(i)
        
        cs.findNextImage()
        image_ids.append(cs.getImageIndex())        
        dn.clearBeliefs()
        for j in range(layers):
            dn.setLayerIsTraining(j, False)
        for j in range(layers):
            dn.setLayerIsTraining(j, True)
            dn.doDestin(cs.getGrayImageFloat())

        for j in range(layers // 2):
            dn.doDestin(cs.getGrayImageFloat())

    #save it
    #dn.save( str(int(time.time()))+"_som.dst")
    dn.save( "saved.dst")


def showDestinImage(i):
    im_id = i % len(image_ids)
    
    cs.setCurrentImage(image_ids[im_id])
    
    dn.clearBeliefs()
    
    for j in range(layers):
        dn.doDestin(cs.getGrayImageFloat())
            
#train the self organizing maps
def train_som():

    global som
    som = pd.Som(64,64, be.getOutputSize() )
    for j in range(layers):
        dn.setLayerIsTraining(j, False)

    for i in range(som_train_iterations):
        if i % 50 == 0:
            print "Training SOM iteration: " + str(i)
        
        showDestinImage(i)

        som.train_iterate( be.getBeliefs() )
        
        if i % 50  == 0:
            som.showSimularityMap()
            
    #finish by saving
    som.saveSom("saved.som")
            
    
def showCifarImage(id):
     cs.setCurrentImage(id)
     hg.cvShowImage("hey", cs.getColorImageMat())
     hg.cvWaitKey()
	

def paintClasses():
    som.clearSimMapMarkers()
    for i in range(len(image_ids)):
        showDestinImage(i)
        label = cs.getImageClassLabel()
        bmu = som.findBestMatchingUnit( be.getBeliefs() )
        hue = label / 10.0
        som.addSimMapMaker(bmu.y, bmu.x, hue, marker_width)
        
    #finish        
    som.showSimularityMap()
        
        
train_destin()
train_som()

paintClasses()
    

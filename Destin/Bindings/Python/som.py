import time
import pydestin as pd

import opencv as cv
import opencv.highgui as hg

cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"
cs = pd.CifarSource(cifar_dir, 2)

#must be size 4 because the cifar data is 32x32
centroids = [5,5,5,5]
layers = 4

training_iterations = 10000

som_train_iterations = 10000

dn = pd.DestinNetworkAlt(pd.W32, layers, centroids)
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

iterations_per_image = 8

marker_width = 6

som_type = pd.ClusterSom



cs.disableAllClasses()
#cs.setClassIsEnabledByName("airplane", True)
#cs.setClassIsEnabledByName("deer", True)
#cs.setClassIsEnabledByName("truck", True)
cs.setClassIsEnabled(0, True)
cs.setClassIsEnabled(5, True)
#cs.setClassIsEnabled(9, True)
# cs.setClassIsEnabled(3, True)

batch = 0

image_ids = []

be = pd.BeliefExporter(dn, 1)
som_width = 64
som_height = 64

sp = None

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

def createSom():
    global som
    som = som_type(som_height,som_width, be.getOutputSize())
    #euclian = e
    #cosine = u
    som.setDistMetric('u')

#train the network
def train_destin():
    global image_ids
    image_ids = []
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

    createSom()
    
    global sp
    sp = pd.SomPresentor(som)
    for j in range(layers):
        dn.setLayerIsTraining(j, False)

    for i in range(som_train_iterations):
                
        showDestinImage(i)

        som.addTrainData(be.getBeliefs() )
        
    som.train(som_train_iterations)
    sp.showSimularityMap()
            
    #finish by saving
    #   som.saveSom("saved.som")
            
    
def showCifarImage(id):
     cs.setCurrentImage(id)
     ci = cs.getColorImageMat()
     
     pd.imshow("Cifar Image: " + str(id), ci)
     hg.cvWaitKey()
	
#blue = 0
#yellow = .5
     
def paintClasses(classes_to_show = []):
    sp.clearSimMapMarkers()
    for i in range(len(image_ids)):
        showDestinImage(i)
        label = cs.getImageClassLabel()
        if len(classes_to_show) == 0 or classes_to_show.count(label) > 0:
            bmu = som.findBestMatchingUnit( be.getBeliefs() )
            hue = label / 10.0
            sp.addSimMapMaker(bmu.y, bmu.x, hue, marker_width)
        
    #finish        
    sp.showSimularityMap()
        

def go():
    train_destin()
    train_som()
    paintClasses()
    

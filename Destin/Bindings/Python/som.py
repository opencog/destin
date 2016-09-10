import pydestin as pd
import cv2.cv as cv
import os
import threading

hg = lambda: None

hg.cvWaitKey = cv.WaitKey
hg.cvSetMouseCallback = cv.SetMouseCallback
"""
This script defines a "go()" function which will train DeSTIN on CIFAR images ( see http://www.cs.toronto.edu/~kriz/cifar.html )
and then presents the DeSTIN beliefs on a self organizing map ( SOM ).

See http://www.mediafire.com/view/?17ehjc28z922g#um21dwtl1lz1f8v for a screen shot of this script.
Each colored dot represents an image. The color of the dot is determined by the image class such as dog or airplane.
The self organizing map should put simular images near each other.

If you click on SOM it will show you CIFAR image of the nearest dot.

"""

# Download the required data at http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
# Set this variable to the folder containing data_batch_1.bin to data_batch_5.bin
cifar_dir = os.getenv("HOME") + "/destin/Destin/Data/cifar-10-batches-bin"

cifar_batch = 1 #which CIFAR batch to use from 1 to 5
cs = pd.CifarSource(cifar_dir, cifar_batch)

#must have 4 layers because the cifar data is 32x32
layers = 4
centroids = [7,5,5,5]

# How many CIFAR images to train destin with. If larger than
# If this this is larger than the number of possible CIFAR images then some
# images will be repeated
training_iterations = 20000

som_train_iterations = 10000

is_uniform = True # uniform DeSTIN or not
dn = pd.DestinNetworkAlt(pd.W32, layers, centroids, is_uniform)

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

#How wide the color dots shows up on the SOM map
marker_width = 6

# which SOM class to use ( currenly only ClusterSom works)
som_type = pd.ClusterSom

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

som_window_title="SOM"

# How many cells wide and high the SOM is
som_width = 64
som_height = 64

# How many pixels wide and high it should be shown on the screen
som_display_width = 512
som_display_height = 512

# 1/2 the width of the neighborhood block used in calculating the
# grayscale simularity map image. A larger width makes the image smoother
# but takes longer to calcuate.
# See the imformation about the grayscale image simularity measurement here http://davis.wpi.edu/~matt/courses/soms/#Quality
som_sim_map_nh_width = 3

# SOM presentor global variable    
sp = None

# which ids of the CIFAR images that were used in training
image_ids = []

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
    be.setBottomLayer(bottom_belief_layer)  # calculates output size
    som = som_type(som_height,som_width, be.getOutputSize())
    #euclian = e
    #cosine = u #TODO: test that the character is surviving the translation into c code
                #and not defaulting back to euclean
    som.setDistMetric('u')

#train the network
def train_destin():
    global image_ids
    image_ids = []
    for i in range(training_iterations):
        if i % 100 == 0:
            print "Training DeSTIN iteration: " + str(i)

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
            dn.doDestin(cs.getGrayImageFloat())
            
        #let it train for 2 more times with all layers training
        for j in range(2):
            dn.doDestin(cs.getGrayImageFloat())

    #
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

    #turn off destin training so
    #its beliefs for a given image stay fixed
    for j in range(layers):
        dn.setLayerIsTraining(j, False)
        
    for i in range(som_train_iterations):
        showDestinImage(i)
        som.addTrainData(be.getBeliefs() )
    som.train(som_train_iterations)

            
    #finish by saving
    #   som.saveSom("saved.som")
            
    
def showCifarImage(id):
     cs.setCurrentImage(id)
     ci = cs.getColorImageMat()
     pd.imshow("Cifar Image: " + str(id), ci)
     
	
#blue = 0
#yellow = .5


# paints the colored image dots on the SOM
coords_to_image_index = None
def paintClasses(classes_to_show = []):
    print "Calculating dot locations..."
    sp.clearSimMapMarkers()
    global coords_to_image_index
    coords_to_image_index = {}
    for i in range(len(image_ids)):
        showDestinImage(i)
        label = cs.getImageClassLabel()
        if len(classes_to_show) == 0 or classes_to_show.count(label) > 0:
            bmu = som.findBestMatchingUnit( be.getBeliefs() )
            hue = label / 10.0
            coords_to_image_index[ ( bmu.x, bmu.y) ] = cs.getImageIndex()
            sp.addSimMapMaker(bmu.y, bmu.x, hue, marker_width)
        
    #finish
    print "Calculating Simularity Grayscale Map"
    sp.showSimularityMap(som_window_title,
                         som_sim_map_nh_width,
                         som_display_width,
                         som_display_height)
    hg.cvSetMouseCallback(som_window_title, som_click_callback)


# This callback lets the user click on a dot on the SOM and it will display
# the corresponding image for it.
def som_click_callback(event, x, y, flag, param):
    if event == cv.CV_EVENT_LBUTTONUP:
        print "clicked r:%i c:%i" % (y, x)
        scale_x = som_width /  float(som_display_width)
        scale_y = som_height / float(som_display_height)
        minDist = 1e100

        # Iterate through all the best matching unit coordinates
        #to find which map marker is closest to the click
        for bmu_coords in coords_to_image_index:
            #convert mouse click coordinates to SOM coordinates
            xx = x * scale_x
            yy = y * scale_y
            bmu_x = bmu_coords[0]
            bmu_y = bmu_coords[1]
            dist = abs(bmu_x - xx) + abs(bmu_y - yy)
            if dist < minDist:
                minDist = dist
                min_bmu_coords = bmu_coords

        image_index = coords_to_image_index[min_bmu_coords]
        print "BMU X: %i Y:%i, Image index: %i" % (min_bmu_coords[0], min_bmu_coords[1], image_index )
        cs.displayColorImage(image_index)
        cs.displayGrayImage(image_index)
                
                

# This waitkey thread lets the opencv windows refresh automatically
# without needed to manually call the cv::waitkey method
kill_waitkey = False
class waitkey(threading.Thread):
    def run(self):
        while True:    
            hg.cvWaitKey(100)
            if kill_waitkey:
                print "waitkey killed"
                return

waitkey().start()


def go():
    train_destin()
    print "Training SOM..."
    train_som()
    paintClasses()
    print "Done."

#Start it all up
go()

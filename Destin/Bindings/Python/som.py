import time
import pydestin as pd
import opencv as cv
import opencv.highgui as hg
import threading


cifar_dir = "/home/ted/destin_git_repo/Destin/Data/CIFAR/cifar-10-batches-bin"

cifar_batch = 1 #from 1 to 4
cs = pd.CifarSource(cifar_dir, cifar_batch)

#must be size 4 because the cifar data is 32x32
centroids = [5,5,5,5]
layers = 4


bottom_belief_layer = 1 # The som trains on concatenated beliefs starting from this layer to the top layer


training_iterations = 10000

som_train_iterations = 1000

dn = pd.DestinNetworkAlt(pd.W32, layers, centroids)
be = pd.BeliefExporter(dn, bottom_belief_layer)
dn.setParentBeliefDamping(0)
dn.setPreviousBeliefDamping(0)

iterations_per_image = 8

marker_width = 6

som_type = pd.ClusterSom

cs.disableAllClasses()
#cs.setClassIsEnabledByName("airplane", True)

cs.setClassIsEnabled(0, True)
cs.setClassIsEnabled(5, True)

image_ids = []




som_window_title="SOM"
som_width = 64
som_height = 64
som_display_width = 512
som_display_height = 512
som_sim_map_nh_width = 2

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
        if i % 100 == 0:
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

            
    #finish by saving
    #   som.saveSom("saved.som")
            
    
def showCifarImage(id):
     cs.setCurrentImage(id)
     ci = cs.getColorImageMat()
     pd.imshow("Cifar Image: " + str(id), ci)
     
	
#blue = 0
#yellow = .5

coords_to_image_index = None
def paintClasses(classes_to_show = []):
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
    sp.showSimularityMap(som_window_title,
                         som_sim_map_nh_width,
                         som_display_width,
                         som_display_height)
    hg.cvSetMouseCallback(som_window_title, som_click_callback)


def som_click_callback(event, x, y, flag, param):
    if event == hg.CV_EVENT_LBUTTONUP:
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
        cs.displayCifarColorImage(image_index)
        cs.displayCifarGrayImage(image_index)
                
                

kill_waitkey = False
class waitkey(threading.Thread):
    def run(self):
        while True:    
            hg.cvWaitKey(100)
            if kill_waitkey:
                print "waitkey killed"
                return

def go():
    train_destin()
    train_som()
    paintClasses()


waitkey().start()


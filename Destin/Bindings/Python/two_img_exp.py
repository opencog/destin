import pydestin as pd
import cv2.cv as cv
import time

ims = pd.ImageSouceImpl()

ims.addImage("/home/ted/Pictures/I.png")
ims.addImage("/home/ted/Pictures/X.png")
ims.addImage("/home/ted/Pictures/Y.png")

centroids = [2,2,8,32,64,32,16,3]  
dn = pd.DestinNetworkAlt( pd.W512, 8, centroids, True)

def train():
    for i in range(3200):
        if i % 10 == 0:
            print "Iteration " + str(i)
            
        ims.findNextImage()
        #dn.clearBeliefs()
        for j in range(2):
            
            f = ims.getGrayImageFloat()    
            dn.doDestin(f)
 
def dci(layer, cent, equalize_hist = False, exp_weight = 4):
    dn.setCentImgWeightExponent(exp_weight)
    dn.displayCentroidImage(layer, cent, 512, equalize_hist)
    cv.WaitKey(100)



def window_callback(event, x, y, flag, param):
    if event == cv.CV_EVENT_LBUTTONUP:
        pass
#dn.load("x.dst")
#cv.SetMouseCallback("Centroid image",window_callback)

do_train = True
if do_train:
    train()
    t = str(int(time.time()))
    fn = t + ".dst"
    print "Saving " + fn
    dn.save(fn)
else:
    to_load = "1357965378.dst"
    dn.load(to_load)
dci(7,0,False, 4)
 
import pydestin as pd
import cv2.cv as cv

ims = pd.ImageSouceImpl()

ims.addImage("/home/ted/Pictures/I.png")
ims.addImage("/home/ted/Pictures/X.png")

centroids = [2,2,8,8,64,16,4,2]
dn = pd.DestinNetworkAlt( pd.W512, 8, centroids, True)

def train():
    for i in range(800):
        if i % 10 == 0:
            print "Iteration " + str(i)
            
        ims.findNextImage()
        #dn.clearBeliefs()
        for j in range(8):
            
            f = ims.getGrayImageFloat()    
            dn.doDestin(f)
 
def dci(layer, cent, equalize_hist = False, exp_weight = 4):
    dn.setCentImgWeightExponent(exp_weight)
    dn.displayCentroidImage(layer, cent, 512, equalize_hist)
    cv.WaitKey(100)
    
#dn.load("x.dst")
train()
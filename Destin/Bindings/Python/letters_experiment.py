import sys
import os, errno
import cv2.cv as cv
import time
import pydestin as pd
import czt_mod as czm
import charting as chart

batch_mode = False
experiment_id = None

# print usage for batch mode
def usage():
    print "Usage: " + sys.argv[0] + " <experiment_id> <freq_coeff> <freq_treshold> <add_coeff> " \
            + "<starv_coeff> <start_centroids> <max_centroids>"
    sys.exit(1)

if (len(sys.argv) > 1):
    try:
        batch_mode = True
        start_time = time.time()
        experiment_id = sys.argv[1]
        freq_coeff = float(sys.argv[2])
        freq_treshold = float(sys.argv[3])
        add_coeff = float(sys.argv[4])
        starv_coeff = float(sys.argv[5])
        start_centroids = int(sys.argv[6])
        max_centroids = int(sys.argv[7])
    except:
        usage()

experiment_root_dir="./experiment_runs"

ims = pd.ImageSouceImpl(512, 512)

#letters = "LO+"
letters = "ABCDE"
for l in letters:
    ims.addImage(czm.homeFld + "/Downloads/destin_toshare/train images/%s.png" % l)

if not batch_mode:
    centroids = [4,8,16,32,64,32,16,len(letters)]
else:
    centroids = [start_centroids for i in range(7)]
    centroids.append(len(letters))

layers = len(centroids)
top_layer = layers - 1
draw_layer = top_layer
iterations = 3000
#image_mode = pd.DST_IMG_MODE_RGB
image_mode = pd.DST_IMG_MODE_GRAYSCALE
dn = pd.DestinNetworkAlt( pd.W512, layers, centroids, True, None, image_mode)
dn.setFixedLearnRate(.1)
dn.setBeliefTransform(pd.DST_BT_NONE)
pd.SetLearningStrat(dn.getNetwork(), pd.CLS_FIXED)

if batch_mode:
    dn.setFrequencyCoefficients(freq_coeff, freq_treshold, add_coeff)
    dn.setStarvationCoefficient(starv_coeff)
    dn.setMaximumCentroidCounts(max_centroids)

#dn.setBeliefTransform(pd.DST_BT_P_NORM)
#ut=1.5
#dn.setTemperatures([ut,ut,ut,ut,ut,ut,ut,ut])

weight_exponent = 4

save_root="./saves/"


def train():
    for i in xrange(iterations):
        if i % 10 == 0:
            variances = dn.getLayersVariances();
            separations = dn.getLayersSeparations();

            chart.update([separations[draw_layer] - variances[draw_layer], variances[draw_layer], separations[draw_layer]])
            chart.draw()
            
            if (i % 300) == 0:
                print "#####################################";
                print "Iteration " + str(i)
                print ""

                for j in range(len(centroids)):
                    print "Layer: " + str(j)
                    print "Variance: " + str(variances[j])
                    print "Separation: " + str(separations[j])
                    print "Quality: " + str(separations[j] - variances[j])
                    print "Centroids: " + str(dn.getBeliefsPerNode(j))
                    print ""
                sys.stdout.flush()

        ims.findNextImage()
        #dn.clearBeliefs()
        for j in range(1):
            if image_mode == pd.DST_IMG_MODE_GRAYSCALE:
                f = ims.getGrayImageFloat()    
            elif image_mode == pd.DST_IMG_MODE_RGB:
                f = ims.getRGBImageFloat()
            else:
                raise Exception("unsupported image mode")
            dn.doDestin(f)

#display centroid image
def dci(layer, cent, equalize_hist = False, exp_weight = 4):
    if cent >= dn.getBeliefsPerNode(layer):
        print "centroid out bounds"
        return
    dn.setCentImgWeightExponent(exp_weight)
    dn.displayCentroidImage(layer, cent, 256, equalize_hist)
    cv.WaitKey(100)

def dcis(layer):
    dn.displayLayerCentroidImages(layer,1000)
    cv.WaitKey(100)
    

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def saveCenImages(run_id, layer):
    run_dir = experiment_root_dir + "/"+run_id+"/"
    mkdir(run_dir)

    # with centroid weight = 1 
    orig_dir = run_dir+"orig/"
    
    # with centroid weight = 1 and enhanced = true
    orige_dir = run_dir+"orig_e/"
    
    # store centroid with higher weighting
    highweighted_dir = run_dir+"highweight/"
    
    highweightede_dir = run_dir+"highweight_e/"

    for d in [orig_dir, orige_dir, highweighted_dir, highweightede_dir]:
        mkdir(d)
    
    # Save centriod images
    bn = dn.getBeliefsPerNode(layer)
        
    dn.setCentImgWeightExponent(1)
    dn.updateCentroidImages()
    for i in range(bn):
        f = "%02d_%02d_%s.png" % (layer, i,run_id)
        # save original
        fn = orig_dir + f 
        dn.saveCentroidImage(layer, i, fn, 512, False )
    
        # save original enhanced
        fn = orige_dir + f
        dn.saveCentroidImage(layer, i, fn, 512, True )
    
    
    dn.setCentImgWeightExponent(weight_exponent)
    dn.updateCentroidImages()
    for i in range(bn):
        f = "%02d_%02d_%s.png" % (layer, i,run_id)
        fn = highweighted_dir + f 
        dn.saveCentroidImage(layer, i, fn, 512, False )
        fn = highweightede_dir + f
        dn.saveCentroidImage(layer, i, fn, 512, True )

def print_csv_entry():
    variances = dn.getLayersVariances();
    separations = dn.getLayersSeparations();

    entries = [str(experiment_id), str(freq_coeff), str(freq_treshold), str(add_coeff), \
               str(starv_coeff), str(len(letters)), str(iterations)]

    for j in range(len(centroids)):
        entries.append(str(dn.getBeliefsPerNode(j)))
        entries.append(str(variances[j]))
        entries.append(str(separations[j]))
        entries.append(str(separations[j] - variances[j]))
    entries.append(str(time.time() - start_time))

    separator = ","
    print "CSV: " + separator.join(entries)

do_train = True
save = True
if do_train:
    train()
    if save:    
        t = str(int(time.time()))
        if experiment_id is not None:
            t += "_" + experiment_id
        fn = save_root + t + ".dst"
        print "Saving " + fn
        dn.save(fn)
        for i in range(layers):
            saveCenImages(t,i)
        if batch_mode:
            print_csv_entry()
else:
    to_load = "letter_exp.dst"
    dn.load(to_load)

dn.save("letter_exp.dst")
dcis(top_layer)


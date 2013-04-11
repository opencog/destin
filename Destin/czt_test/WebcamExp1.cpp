#include <time.h>

#include "VideoSource.h"
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "stdio.h"
#include "unit_test.h"

using namespace cv;
void testNan(float * array, int len){
    for(int i = 0 ; i < len ; i++){
        if(isnan(array[i])){
            printf("input had nan\n");
            exit(1);
        }
    }
}

/** meaures time between calls and prints the fps.
 *  print - prints the fps if true, otherwise just keeps the timer consistent
 */
void printFPS(bool print){
    // start = initial frame time
    // end = final frame time
    // sec = time count in seconds
    // set all to 0
    static double end, sec, start = 0;

    // set final time count to current tick count
    end = (double)cv::getTickCount();

    //
    if(start != 0){
        sec = (end - start) / getTickFrequency();
        if(print==true){
            printf("fps: %f\n", 1 / sec);
        }
    }
    start = (double)cv::getTickCount();
}

int main(int argc, char ** argv){

    //VideoSource vs(false, "./Various.avi");
    VideoSource vs(true, "");

    vs.enableDisplayWindow();

    SupportedImageWidths siw = W512;

    // Left to Right is bottom layer to top
    //uint centroid_counts[]  = {3,3,4,3,3,3,3,5};
    uint centroid_counts[]  = {4,3,5,3,3,2,3,4};
    bool isUniform = true;

    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    while(vs.grab()){

        frameCount++;


        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card
        testNan(t.getDest(), 512*512);

        network->doDestin(t.getDest());

        if(frameCount % 2 != 0 ){ //only print every 2rd so display is not so jumpy
            printFPS(false);
            continue;
        }

        // Old clear screen method
        //printf("\033[2J");

        // New clear screen method (might give less flickering...?)
        printf("\033[2J\033[1;1H");

        printf("Frame: %i\n", frameCount);
        printFPS(true);
        int layer = 1;
        Node & n = *network->getNode(layer,0,0);
        printf("Node %i,0,0 winner: %i\n",layer, n.winner);
        printf("Node centroids: %i\n", n.nb);

        printf("Node starv:");
        printFloatArray(n.starv, n.nb);
        printf("Starv coef: %f \n", n.starvCoeff);
        printf("\n");

        // 2013.4.5
        // CZT
        //
        //printf("layer %i node 0 centroid locations:\n", layer);
        //network->printNodeCentroidPositions(layer, 0, 0);
        for(int l = 0 ; l < 8 ; l++){
            printf("belief graph layer: %i\n",l);
            network->printBeliefGraph(l,0,0);
        }

    }

    delete network;
    return 0;
}

#include <time.h>

#include "VideoSource.h"
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "stdio.h"
#include "unit_test.h"

extern "C"{
#define UINT64_C //hack to avoid compile error
#include <libavutil/log.h> //used to turn off opencv warning message

}

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
    static double end, sec, start = 0;

    end = (double)cv::getTickCount();
    if(start != 0){
        sec = (end - start) / getTickFrequency();
        if(print==true){
            printf("fps: %f\n", 1 / sec);
        }
    }
    start = (double)cv::getTickCount();
}

int main(int argc, char ** argv){

    av_log_set_level(AV_LOG_QUIET);//turn off message " No accelerated colorspace conversion found from yuv422p to bgr24"

    //VideoSource vs(false, "./destin_video_test.avi");
    //VideoSource vs(false, "./cowboy.avi");
    VideoSource vs(true, "");

    vs.enableDisplayWindow();

    SupportedImageWidths siw = W512;

    uint centroid_counts[]  = {9,8,7,7,5,4,3,2};

    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    float * beliefs;

    while(vs.grab()){

        frameCount++;


        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card
        testNan(t.getDest(), 512*512);

        network->doDestin(t.getDest());

        if(frameCount % 1 != 0 ){ //only print every 3rd so display is not so jumpy
            printFPS(false);
            continue;
        }

        
        printf("\033[2J");      /* clear screen */
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

        printf("layer %i node 0 centroid locations:\n", layer);
        network->printNodeCentroidPositions(layer, 0, 0);
        for(int l = 0 ; l < 8 ; l++){
            printf("belief graph layer: %i\n",l);
            network->printBeliefGraph(l,0,0);
        }

    }

    delete network;
    return 0;
}

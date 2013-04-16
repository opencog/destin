/*
2013.4.5 VERSION

I want to re-do what I thought again!!!
*/
#include "stdio.h"
#include "VideoSource.h"
#include "DestinNetworkAlt.h"
#include "Transporter.h"
#include "unit_test.h"
#include <time.h>
#include "macros.h"

//#define TEST_ORG
//#define TEST_IN_ORG
#define TEST_STEP1

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
double printFPS(bool print){
    // start = initial frame time
    // end = final frame time
    // sec = time count in seconds
    // set all to 0
    static double end, sec, start = 0;

    // set final time count to current tick count
    end = (double)cv::getTickCount();

    float out;
    //
    if(start != 0){
        sec = (end - start) / getTickFrequency();
        if(print==true){
            printf("fps: %f\n", 1 / sec);
        }
        out = 1/sec;
    }
    start = (double)cv::getTickCount();
    return out;
}

/*
  2013.4.9
*/
void combineWithDepth_1(float * fIn, int size, int extRatio, float * tempOut)
{
    int i,j;
    for(i=0; i<size; ++i)
    {
        tempOut[i] = fIn[i];
    }
    for(i=1; i<extRatio; ++i)
    {
        for(j=0; j<size; ++j)
        {
            //tempOut[size*i+j] = (float)rand() / (float)RAND_MAX;
            tempOut[size*i+j] = 0.5;
        }
    }
}

int main(int argc, char ** argv)
{

/*****************************************************************************/
/*
  2013.4.5

  I want to test the original destin codes again and learn the CMake further!
*/
#ifdef TEST_ORG
    VideoSource vs(true, "");

    vs.enableDisplayWindow();

    SupportedImageWidths siw = W512;

    // Left to Right is bottom layer to top
    // CZT
    // From whole image level to small square level
    //
    //uint centroid_counts[]  = {3,2,3,2,3,2,3,4};
    uint centroid_counts[]  = {4,3,5,3,3,2,3,4};
    bool isUniform = true;

    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
#ifdef TEST_IN_ORG
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, 512*512, 1);
#endif

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    double totalFps = 0.0;
    while(vs.grab()){

        frameCount++;


        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card
        testNan(t.getDest(), 512*512);

        network->doDestin(t.getDest());

        if(frameCount % 2 != 0 ){ //only print every 2rd so display is not so jumpy
            totalFps += printFPS(false);
            continue;
        }

        // Old clear screen method
        //printf("\033[2J");

        // New clear screen method (might give less flickering...?)
        printf("\033[2J\033[1;1H");
        printf("----------------TEST_ORG----------------\n");

        printf("Frame: %i\n", frameCount);
        totalFps += printFPS(true);
        printf("Average FPS now: %f\n", totalFps/frameCount);
        //int layer = 1;
        // CZT
        //
        int layer = 7;
        Node & n = *network->getNode(layer,0,0);
        printf("Node %i,0,0 winner: %i\n",layer, n.winner);
        printf("Node centroids: %i\n", n.nb);

        printf("Node starv:");
        printFloatArray(n.starv, n.nb);
        printf("Starv coef: %f \n", n.starvCoeff);
        printf("\n");

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
#endif

/*****************************************************************************/
/*
  2013.4.5
  Step 1: I want to add the struct, but only use pixelValue and keep the original algorithm
  running correctly;

  2013.4.8
  Step 2: Modify DavisDestin's CMake file;

  2013.4.9
  Step 3: Change to a new method, float *;

  2013.4.10
  Step 4: 'extRatio';
*/
#ifdef TEST_STEP1
    VideoSource vs(true, "");
    vs.enableDisplayWindow();
    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,3,5,3,3,2,3,3};
    bool isUniform = true;

    // 2013.4.10
    // CZT
    //
    int size = 512*512;
    int extRatio = 2;
    int inputSize = size*extRatio;
    float * tempIn;
    MALLOC(tempIn, float, inputSize); //

    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    double totalFps = 0.0;
    while(vs.grab()){
        frameCount++;

        t.setSource(vs.getOutput());
        t.transport(); //move video from host to card
        testNan(t.getDest(), 512*512);

        // CZT
        //
        //network->doDestin(t.getDest());

        // Method2:
        combineWithDepth_1(t.getDest(), size, extRatio, tempIn);
        network->doDestin_c1(tempIn);
        //break;

        if(frameCount % 2 != 0 ){ //only print every 2rd so display is not so jumpy
            totalFps += printFPS(false);
            continue;
        }

        // Clean screen
        printf("\033[2J\033[1;1H");
        printf("----------------TEST_STEP1----------------\n");

        printf("Frame: %i\n", frameCount);
        totalFps += printFPS(true);
        printf("Average FPS now: %f\n", totalFps/(frameCount-1));
        // CZT
        //
        //int layer = 1;
        //int layer = 0;
        int layer = 7;
        Node & n = *network->getNode(layer,0,0);
        printf("Node %i,0,0 winner: %i\n",layer, n.winner);
        printf("Node centroids: %i\n", n.nb);

        printf("Node starv:");
        printFloatArray(n.starv, n.nb);
        printf("Starv coef: %f \n", n.starvCoeff);
        printf("\n");

        // CZT
        //
        //printf("layer %i node 0 centroid locations:\n", layer);
        //network->printNodeCentroidPositions(layer, 0, 0);
        for(int l = 0 ; l < 8 ; l++){
            printf("belief graph layer: %i\n",l);
            network->printBeliefGraph(l,0,0);
        }
    }/**/

    /*// 2013.4.9
    // CZT
    // I want to see the detailed struct for Destin and Node.
    //
    int i;
    for(i=0; i<8; ++i)
    {
        printf("Layer %d has %d nodes!\n", i, network->getNetwork()->layerSize[i]);
    }*/

    /*// 2013.4.10
    // CZT
    //
    int ni=4*4;
    int i;
    for(i=0; i<ni; ++i)
    {
        printf("OffSet: %d\n", network->getNode(0, 0, 0)->inputOffsets[i]);
    }*/

    FREE(tempIn);
    delete network;
#endif

	return 0;
}

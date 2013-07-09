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
//
#include "stereovision.h"
#include "stereocamera.h"
#include "czt_lib2.h"
#include "czt_lib.h"

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
            tempOut[size*i+j] = (float)rand() / (float)RAND_MAX;
            //tempOut[size*i+j] = 0.5;
        }
    }
}

void combineWithDepth_2(float * fIn, float * depth, int size, float * tempOut)
{
    int i,j;
    for(i=0; i<size; ++i)
    {
        tempOut[i] = fIn[i];
    }
    for(i=0; i<size; ++i)
    {
        tempOut[size+i] = depth[i];
    }
}

IplImage * convert_c2(cv::Mat in)
{
    IplImage * out = new IplImage(in);
    IplImage * real_out;
    real_out = cvCreateImage(cvGetSize(out), IPL_DEPTH_8U, 1);
    cvCvtColor(out, real_out, CV_BGR2GRAY);
    return real_out;
}

void convert(cv::Mat & in, float * out) {
    if(in.channels()!=1){
        throw runtime_error("Excepted a grayscale image with one channel.");
    }
    if(in.depth()!=CV_8U){
        throw runtime_error("Expected image to have bit depth of 8bits unsigned integers ( CV_8U )");
    }
    cv::Point p(0, 0);
    int i = 0 ;
    for (p.y = 0; p.y < in.rows; p.y++) {
        for (p.x = 0; p.x < in.cols; p.x++) {
            //i = frame.at<uchar>(p);
            //use something like frame.at<Vec3b>(p)[channel] in case of trying to support color images.
            //There would be 3 channels for a color image (one for each of r, g, b)
            out[i] = (float)in.at<uchar>(p) / 255.0f;
            i++;
        }
    }
}

int main(int argc, char ** argv)
{

//#define TEST_ORG
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

//#define TEST_STEP1
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

  2013.7.5
  TODO: the following codes could be removed; just the tests in the beginning;
*/
/*#ifdef TEST_STEP1
    //VideoSource vs(false, "./Various.avi");
    VideoSource vs(true, "");
    vs.enableDisplayWindow();
    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,3,5,3,3,2,3,3};
    bool isUniform = true;

    // 2013.4.10
    // CZT
    //
    int size = 512*512;
    int extRatio = 3; // Use this parameter to control the size!!!

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

        // Clean screencvRemap
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
    }

    // 2013.4.9
    // CZT
    // I want to see the detailed struct for Destin and Node.
    //
    int i;
    for(i=0; i<8; ++i)
    {
        printf("Layer %d has %d nodes!\n", i, network->getNetwork()->layerSize[i]);
    }

    // 2013.4.10
    // CZT
    //
    int ni=4*4;
    int i;
    for(i=0; i<ni; ++i)
    {
        printf("OffSet: %d\n", network->getNode(0, 0, 0)->inputOffsets[i]);
    }

    FREE(tempIn);
    delete network;
#endif*/

//#define TEST_STEP2
/*****************************************************************************/
/*
  2013.4.19
  Try to use 2 cams!!!

  2013.7.5
  TODO: the following codes could be referred as how to use 2-webcam input;
  could be removed sometime;
*/
#ifdef TEST_STEP2
    VideoSource vs1(true, "", CV_CAP_ANY);
    VideoSource vs2(true, "", CV_CAP_ANY+1);
    vs1.enableDisplayWindow_c1("left");
    vs2.enableDisplayWindow_c1("right");
    vs1.grab();
    vs2.grab();/**/

    int result;
    StereoVision * sv = new StereoVision(640, 480);
    //StereoVision * sv = new StereoVision(512, 512);
    result = sv->calibrationLoad("calibration.dat");
    printf("calibrationLoad() status: %d\n", result);
    CvSize imageSize = sv->getImageSize();
    CvMat * imageRectifiedPair = cvCreateMat( imageSize.height, imageSize.width*2,CV_8UC3 );
    IplImage * img_l;
    IplImage * img_r;
    float * float_l, * float_r, * float_depth, * float_combined;
    MALLOC(float_l, float, 512*512);
    MALLOC(float_r, float, 512*512);
    MALLOC(float_depth, float, 512*512);
    MALLOC(float_combined, float, 512*512*2);

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,3,5,3,3,2,3,3};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 2;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);
    int frameCount = 0;
    double totalFps = 0.0;

    // Method 1
    // Should use 'VideoSource' part!!!
    //
    while(vs1.grab() && vs2.grab())
    {
        frameCount++;

        img_l = convert_c2(vs1.getOutput_c1());
        img_r = convert_c2(vs2.getOutput_c1());
        result = sv->stereoProcess(img_l, img_r);
        cvShowImage("depth", sv->imageDepthNormalized_c1);
        //printf("%d * %d\n", vs1.getOutput_c1().rows, vs1.getOutput_c1().cols);
        //printf("%d * %d\n", img_l->width, img_l->height);
        //printf("%d * %d\n", sv->imageDepthNormalized->width, sv->imageDepthNormalized->height);

        /*CvMat part;
        cvGetCols( imageRectifiedPair, &part, 0, imageSize.width );
        cvCvtColor( sv->imagesRectified[0], &part, CV_GRAY2BGR );
        cvGetCols( imageRectifiedPair, &part, imageSize.width,imageSize.width*2 );
        cvCvtColor( sv->imagesRectified[1], &part, CV_GRAY2BGR );
        for(int j = 0; j < imageSize.height; j += 16 )
            cvLine( imageRectifiedPair, cvPoint(0,j),cvPoint(imageSize.width*2,j),CV_RGB((j%3)?0:255,((j+1)%3)?0:255,((j+2)%3)?0:255));
        cvShowImage( "rectified", imageRectifiedPair );*/

        float_l = vs1.getOutput();
        float_r = vs2.getOutput();
        cv::Mat tempMat(sv->imageDepthNormalized_c1);
        convert(tempMat, float_depth);
        combineWithDepth_2(float_l, float_depth, 512*512, float_combined);/**/

        network->doDestin_c1(float_combined);
        //network->doDestin_c1(float_l);    // extRatio should be 1 for this!

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
    }/**/

    // Method 2
    // Should comment the 'VideoSourece' Part!!! Here we use 'StereoCamera' class instead!!!
    //
    /*StereoCamera camera;
    if(camera.setup(imageSize) == 0)
    {
        while(true)
        {
            camera.capture();
            img_l = camera.getFramesGray(0);
            img_r = camera.getFramesGray(1);
            cvShowImage("left", img_l);
            cvShowImage("right", img_r);

            sv->stereoProcess(camera.getFramesGray(0), camera.getFramesGray(1));
            cvShowImage("depth", sv->imageDepthNormalized);

            CvMat part;
            cvGetCols( imageRectifiedPair, &part, 0, imageSize.width );
            cvCvtColor( sv->imagesRectified[0], &part, CV_GRAY2BGR );
            cvGetCols( imageRectifiedPair, &part, imageSize.width,imageSize.width*2 );
            cvCvtColor( sv->imagesRectified[1], &part, CV_GRAY2BGR );
            for(int j = 0; j < imageSize.height; j += 16 )
                cvLine( imageRectifiedPair, cvPoint(0,j),cvPoint(imageSize.width*2,j),CV_RGB((j%3)?0:255,((j+1)%3)?0:255,((j+2)%3)?0:255));
            cvShowImage( "rectified", imageRectifiedPair );

            cvWaitKey(10);
        }
    }*/

    cvReleaseImage(&img_l);
    cvReleaseImage(&img_r);
    cvReleaseMat(&imageRectifiedPair);
    FREE(sv);
    FREE(float_l);
    FREE(float_r);
    FREE(float_depth);
    FREE(float_combined);
    delete network;
#endif

//#define TEST_STEP3
/*****************************************************************************/
/*
  2013.7.5
  CZT: to process BGR, 1-webcam video input;
*/
#ifdef TEST_STEP3
    VideoSource vs(true, "");
    vs.enableDisplayWindow();
    vs.turnOnColor();
    czt_lib2 * cl2 = new czt_lib2();

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,3,5,3,3,2,3,4};
    bool isUniform = true;
    bool isExtend = true;
    int nLayers = 8;
    int size = 512*512;
    int extRatio = 3;

    DestinNetworkAlt * network = new DestinNetworkAlt(siw, nLayers, centroid_counts, isUniform, isExtend, size, extRatio);
    float * tempIn;
    MALLOC(tempIn, float, size*extRatio);

    Transporter t;
    vs.grab();//throw away first frame in case its garbage
    int frameCount = 0;

    double totalFps = 0.0;
    while(vs.grab()){
        frameCount++;
        cl2->combineBGR(vs.getBFrame(), vs.getGFrame(), vs.getRFrame(), size, tempIn);
        network->doDestin(tempIn);

        if(frameCount % 2 != 0 ){ //only print every 2rd so display is not so jumpy
            totalFps += printFPS(false);
            continue;
        }

        // New clear screen method (might give less flickering...?)
        printf("\033[2J\033[1;1H");
        printf("----------------TEST_ORG----------------\n");

        printf("Frame: %i\n", frameCount);
        totalFps += printFPS(true);
        printf("Average FPS now: %f\n", totalFps/frameCount);

        int layer = 7;
        Node & n = *network->getNode(layer,0,0);
        printf("Node %i,0,0 winner: %i\n",layer, n.winner);
        printf("Node centroids: %i\n", n.nb);

        printf("Node starv:");
        printFloatArray(n.starv, n.nb);
        printf("Starv coef: %f \n", n.starvCoeff);
        printf("\n");

        for(int l = 0 ; l < 8 ; l++){
            printf("belief graph layer: %i\n",l);
            network->printBeliefGraph(l,0,0);
        }
    }
    delete network;
    FREE(tempIn);
#endif

	return 0;
}

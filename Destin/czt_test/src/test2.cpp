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
//
#include "ImageSourceImpl.h"
#include "czt_lib.h"
#include "czt_lib2.h"

int main(int argc, char ** argv)
{
    /*czt_lib2 * cl2 = new czt_lib2();
    ImageSouceImpl isi;
    isi.addImage("/home/teaera/Downloads/destin_toshare/train images/A.png");

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,8,16,32,64,32,16,8};
    bool isUniform = true;
    int size = 512*512;
    //int extRatio = 2;
    int extRatio = 1;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    int inputSize = size*extRatio;
    float * tempIn;
    MALLOC(tempIn, float, inputSize);

    int frameCount = 0;
    int maxCount = 3000;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
        }

        isi.findNextImage();
        cl2->combineInfo_extRatio(isi.getGrayImageFloat(), size, extRatio, tempIn);
        network->doDestin_c1(tempIn);
    }

    network->displayLayerCentroidImages(7, 1000);
    cv::waitKey(10000);
    //network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.10_A_addRandom.jpg");
    network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.10_A.jpg");*/

    //*************************************************************************
    czt_lib2 * cl2 = new czt_lib2();
    czt_lib * cl = new czt_lib();

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,8,16,32,64,32,16,8};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 2;
    //int extRatio = 1;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    int inputSize = size*extRatio;
    float * tempIn1, * tempIn2, * tempIn;
    MALLOC(tempIn1, float, size);
    MALLOC(tempIn2, float, size);
    MALLOC(tempIn, float, inputSize);
    //cl2->combineImgs("/home/teaera/Work/RECORD/2013.5.8/pro_3/1.jpg", "/home/teaera/Work/RECORD/2013.5.8/pro_add_3/1.jpg", tempIn);
    cl->isNeedResize("/home/teaera/Work/RECORD/2013.5.8/pro_3/1.jpg");
    tempIn1 = cl->get_float512();
    cl->isNeedResize("/home/teaera/Work/RECORD/2013.5.8/pro_add_3/1.jpg");
    tempIn2 = cl->get_float512();
    cl2->combineInfo_depth(tempIn1, tempIn2, size, tempIn);

    int frameCount = 0;
    int maxCount = 5000;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
        }

        network->doDestin_c1(tempIn);
    }

    network->displayLayerCentroidImages(7, 1000);
    cv::waitKey(10000);
    network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.10_faceAndResult.jpg");
    //network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.10_result.jpg");/**/

	return 0;
}

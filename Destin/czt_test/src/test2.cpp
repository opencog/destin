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
//*****************************************************************************
// Add Random depth information
// Test czt_lib2 (which is my own library of functions!)

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

//*****************************************************************************
// This is to test the combined information and show the centroids for combined
// information!
//

    /*czt_lib2 * cl2 = new czt_lib2();
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

    //network->displayLayerCentroidImages(7, 1000);
    //cv::waitKey(10000);
    network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.13_level7.jpg");
    network->saveLayerCentroidImages(6, "/home/teaera/Pictures/2013.5.13_level6.jpg");
    network->saveLayerCentroidImages(5, "/home/teaera/Pictures/2013.5.13_level5.jpg");
    network->saveLayerCentroidImages(4, "/home/teaera/Pictures/2013.5.13_level4.jpg");
    network->saveLayerCentroidImages(3, "/home/teaera/Pictures/2013.5.13_level3.jpg");
    network->saveLayerCentroidImages(2, "/home/teaera/Pictures/2013.5.13_level2.jpg");
    network->saveLayerCentroidImages(1, "/home/teaera/Pictures/2013.5.13_level1.jpg");
    network->saveLayerCentroidImages(0, "/home/teaera/Pictures/2013.5.13_level0.jpg");*/

//*****************************************************************************
// To test the number of centroids for each layer!

    /*ImageSouceImpl isi;
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/6.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/7.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/8.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/9.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/14.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/20.jpg");

    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/1.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/2.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/3.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/4.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/5.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/10.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/11.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/12.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/13.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/15.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/16.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/17.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/18.jpg");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/19.jpg");

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {40,80,160,320,320,120,40,10};
    //uint centroid_counts[]  = {80,160,320,640,640,320,160,80};
    //uint centroid_counts[]  = {320,320,640,640,640,320,320,80};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 1;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    int frameCount = 0;
    int maxCount = 2000;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
        }

        isi.findNextImage();
        network->doDestin_c1(isi.getGrayImageFloat());
    }

    network->displayLayerCentroidImages(7, 1000);
    cv::waitKey(5000);
    network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.24_5.jpg");*/

//*****************************************************************************
// For testing some parameters:

    ImageSouceImpl isi;
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/6.jpg");

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {4,8,16,32,32,12,4,1};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 1;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    int frameCount = 0;
    int maxCount = 800;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
        }

        isi.findNextImage();
        network->doDestin_c1(isi.getGrayImageFloat());
    }

    Destin * dn = network->getNetwork();
    int nLayer = dn->nLayers;
    int i,j;
    for(i=0; i<nLayer; ++i)
    {
        printf("Layer %d:\n", i);
        for(j=0; j<centroid_counts[i]; ++j)
        {
            printf("%d\t", dn->uf_persistWinCounts[i][j]);
        }
        printf("\n----------------\n");
    }

	return 0;
}

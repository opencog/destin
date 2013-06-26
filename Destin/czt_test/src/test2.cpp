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
    //network->saveLayerCentroidImages(7, "/home/teaera/Pictures/2013.5.10_A.jpg");*/

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

/*//*****************************************************************************
// For testing some parameters:

    ImageSouceImpl isi;
    isi.addImage("/home/teaera/Work/RECORD/2013.5.21/downloads_pro_mine/14.jpg");

    SupportedImageWidths siw = W512;
    uint centroid_counts[]  = {2,4,6,8,6,4,2,1};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 1;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    int frameCount = 0;
    int maxCount = 500;
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
            printf("%ld    ", dn->uf_persistWinCounts[i][j]);
        }
        printf("\n----------------\n");
    }

    //for(i=0; i<dn->nBeliefs; ++i)
    //{
    //  printf("%f\n",dn->belief[i]);
    //}
    //network->displayLayerCentroidImages(7, 1000);
    //cv::waitKey(10000);*/

#define TEST_2013_5_30
//#define TEST_ADD
//#define RUN_BEFORE
#define RUN_NOW
//#define SHOW_BEFORE
#define SHOW_NOW
#define TEST_nb
//#define TEST_uf_persistWinCounts
//#define TEST_uf_persistWinCounts_detailed
//#define TEST_uf_avgDelta //uf_sigma
#define TEST_mu
//#define TEST_observation
//#define TEST_beliefMal
//#define TEST_intra
//#define TEST_inter
//#define TEST_VALIDITY

#ifdef TEST_2013_5_30
//*****************************************************************************
    ImageSouceImpl isi;
    //isi.addImage("/home/teaera/Downloads/destin_toshare/train images/A.png");
    isi.addImage("/home/teaera/Work/RECORD/2013.5.8/pro_1/3.jpg");
    czt_lib2 * cl2 = new czt_lib2();
    // currLayer: the layer, in which you want to add centroids or kill centroids;
    // tempLayer: for backup;
    uint tempLayer;
    uint currLayer = 0;
    uint kill_ind = 0;
#define TEST_layer0
#define TEST_layer7

    SupportedImageWidths siw = W512;
    //uint centroid_counts[]  = {1,8,16,32,32,16,8,4};
    //uint centroid_counts[]  = {2,3,4,5,4,3,2,1};
    uint centroid_counts[]  = {6,8,10,12,12,8,6,4};
    bool isUniform = true;
    int size = 512*512;
    int extRatio = 2;
    DestinNetworkAlt * network = new DestinNetworkAlt(siw, 8, centroid_counts, isUniform);
    network->reinitNetwork_c1(siw, 8, centroid_counts, isUniform, size, extRatio);

    float * tempIn;
    MALLOC(tempIn, float, size*extRatio);
    int frameCount;
    int maxCount = 3000;

#ifdef RUN_BEFORE
    frameCount = 0;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
#ifdef TEST_VALIDITY
            for(int i=0; i<8; ++i)
            {
                printf("%e ", network->getValidity(i));
            }
            printf("\n");
#endif // TEST_VALIDITY
        }

        isi.findNextImage();
        //network->doDestin_c1(isi.getGrayImageFloat());
        cl2->combineInfo_extRatio(isi.getGrayImageFloat(), size, extRatio, tempIn);
        network->doDestin_c1(tempIn);
    }
#endif // RUN_BEFORE

#ifdef SHOW_BEFORE
    tempLayer = 7;
    //tempLayer = currLayer;
    network->displayLayerCentroidImages(tempLayer, 1000);
    cv::waitKey(3000);
    network->saveLayerCentroidImages(tempLayer, "/home/teaera/Pictures/2013.6.10_nm_before.jpg");
#endif // SHOW_BEFORE

    Destin * d = network->getNetwork();
    Node * node1 = network->getNode(currLayer, 0, 0);
#ifndef TEST_layer0
    Node * node2 = network->getNode(currLayer-1, 0, 0);
#endif
#ifndef TEST_layer7
    Node * node3 = network->getNode(currLayer+1, 0, 0);
#endif
    int i, l, j;

#ifdef TEST_nb
    printf("------------TEST_nb\n");
    for(i=0; i<d->nLayers; ++i)
    {
        Node * tempNode = network->getNode(i, 0, 0);
        printf("Layer %d\n", i);
        printf("%d  %d  %d  %d  %d\n", d->nb[i], tempNode->ni, tempNode->nb, tempNode->np, tempNode->ns);
        printf("------\n");
    }
    printf("\n");
#endif // TEST_nb

#ifdef TEST_uf_persistWinCounts
    printf("------------TEST_uf_persistWinCounts\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            printf("%ld  ", d->uf_persistWinCounts[l][i]);
            // uf_persistWinCounts, long;
            // uf_starv, float;
            // uf_winCounts, uint;
        }
        printf("\n------\n");
    }
    printf("\n");
#endif // TEST_uf_persistWinCounts

#ifdef TEST_uf_persistWinCounts_detailed
    printf("------------TEST_uf_persistWinCounts_detailed\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            printf("%ld  ", d->uf_persistWinCounts_detailed[l][i]);
        }
        printf("\n------\n");
    }
    printf("\n");
#endif // TEST_uf_persistWinCounts_detailed

#ifdef TEST_uf_avgDelta
    printf("------------TEST_uf_avgDelt\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            for(j=0; j<network->getNode(l,0,0)->ns; ++j)
            {
                //printf("%f  ", d->uf_avgDelta[l][i*network->getNode(l,0,0)->ns+j]);
                printf("%f  ", d->uf_sigma[l][i*network->getNode(l,0,0)->ns+j]);
            }
            printf("\n");
        }
        printf("------\n");
    }
    printf("\n");
#endif // TEST_uf_avgDelta

#ifdef TEST_mu
    printf("------------TEST_mu\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        for(j=0; j<node1->ns; ++j)
        {
            printf("%f  ", node1->mu[i*node1->ns+j]);
        }
        printf("\n---\n");
    }
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        for(j=0; j<node2->ns; ++j)
        {
            printf("%f  ", node2->mu[i*node2->ns+j]);
        }
        printf("\n---\n");
    }
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        for(j=0; j<node3->ns; ++j)
        {
            printf("%f  ", node3->mu[i*node3->ns+j]);
        }
        printf("\n---\n");
    }
#endif
    printf("\n");
#endif // TEST_mu

#ifdef TEST_observation
    printf("------------TEST_observation\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        for(j=0; j<node1->ns; ++j)
        {
            printf("%f  ", node1->observation[i*node1->ns+j]);
        }
        printf("\n---\n");
    }
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        for(j=0; j<node2->ns; ++j)
        {
            printf("%f  ", node2->observation[i*node2->ns+j]);
        }
        printf("\n---\n");
    }
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        for(j=0; j<node3->ns; ++j)
        {
            printf("%f  ", node3->observation[i*node3->ns+j]);
        }
        printf("\n---\n");
    }
#endif
    printf("\n");
#endif // TEST_observation

#ifdef TEST_beliefMal
    printf("------------TEST_beliefMal\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        printf("%f  ", node1->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        printf("%f  ", node2->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        printf("%f  ", node3->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#endif
    printf("\n");
#endif // TEST_beliefMal

#ifdef TEST_intra
    printf("------------TEST_intra\n");
    /*float * variance = network->getVariance(currLayer);
    float * weight = network->getWeight(currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        printf("%f  ", variance[i]);
    }
    printf("\n");
    for(i=0; i<node1->nb; ++i)
    {
        printf("%f  ", weight[i]);
    }
    printf("\n");*/
    printf("%f\n", network->getIntra(currLayer));
    printf("\n");
#endif

#ifdef TEST_inter
    printf("------------TEST_inter\n");
    printf("%f\n", network->getInter(currLayer));
    printf("\n");
#endif

    //printf("%e\n", network->getIntra(currLayer)/network->getInter(currLayer));
//---------------------------------------------------------------------------//
    printf("--------------------------------------------------------------\n\n");
//---------------------------------------------------------------------------//

#ifdef TEST_ADD
    // Add
    centroid_counts[currLayer]++;
    network->updateDestin_add(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer);
    centroid_counts[currLayer]++;
    network->updateDestin_add(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer);
    centroid_counts[currLayer]++;
    network->updateDestin_add(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer);/**/

    // Kill
    /*centroid_counts[currLayer]--;
    network->updateDestin_kill(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer, kill_ind);
    centroid_counts[currLayer]--;
    network->updateDestin_kill(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer, kill_ind);
    centroid_counts[currLayer]--;
    network->updateDestin_kill(siw, 8, centroid_counts, isUniform, size, extRatio, currLayer, kill_ind);*/

#ifdef RUN_NOW
    frameCount = 0;
    while(frameCount < maxCount){
        frameCount++;
        if(frameCount % 10 == 0)
        {
            printf("Count %d;\n", frameCount);
        }

        isi.findNextImage();
        //network->doDestin_c1(isi.getGrayImageFloat());
        cl2->combineInfo_extRatio(isi.getGrayImageFloat(), size, extRatio, tempIn);
        network->doDestin_c1(tempIn);
    }
#endif // RUN_NOW

#ifdef SHOW_NOW
    tempLayer = 7;
    //tempLayer = currLayer;
    network->displayLayerCentroidImages(tempLayer, 1000);
    cv::waitKey(3000);
    network->saveLayerCentroidImages(tempLayer, "/home/teaera/Pictures/2013.6.10_nm_now.jpg");
#endif // SHOW_NOW

    // I don't why I shoud 'reload' these nodes again?
    // Otherwise, the display of node->mu will have some problems?
    // Maybe my fault somewhere?
    //
    node1 = network->getNode(currLayer, 0, 0);
#ifndef TEST_layer0
    node2 = network->getNode(currLayer-1, 0, 0);
#endif
#ifndef TEST_layer7
    node3 = network->getNode(currLayer+1, 0, 0);
#endif

#ifdef TEST_nb
    printf("------TEST_nb\n");
    for(i=0; i<d->nLayers; ++i)
    {
        Node * tempNode = network->getNode(i, 0, 0);
        printf("Layer %d\n", i);
        printf("%d  %d  %d  %d  %d\n", d->nb[i], tempNode->ni, tempNode->nb, tempNode->np, tempNode->ns);
        printf("------\n");
    }
    printf("\n");
#endif // TEST_nb

#ifdef TEST_uf_persistWinCounts
    printf("------TEST_uf_persistWinCounts\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            printf("%ld  ", d->uf_persistWinCounts[l][i]);
            // uf_persistWinCounts, long;
            // uf_starv, float;
            // uf_winCounts, uint;
        }
        printf("\n------\n");
    }
    printf("\n");
#endif // TEST_uf_persistWinCounts

#ifdef TEST_uf_persistWinCounts_detailed
    printf("------------TEST_uf_persistWinCounts_detailed\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            printf("%ld  ", d->uf_persistWinCounts_detailed[l][i]);
        }
        printf("\n------\n");
    }
    printf("\n");
#endif // TEST_uf_persistWinCounts_detailed

#ifdef TEST_uf_avgDelta
    printf("------------TEST_uf_avgDelt\n");
    for(l=0; l<d->nLayers; ++l)
    {
        printf("Layer %d\n", l);
        for(i=0; i<d->nb[l]; ++i)
        {
            for(j=0; j<network->getNode(l,0,0)->ns; ++j)
            {
                //printf("%f  ", d->uf_avgDelta[l][i*network->getNode(l,0,0)->ns+j]);
                printf("%f  ", d->uf_sigma[l][i*network->getNode(l,0,0)->ns+j]);
            }
            printf("\n");
        }
        printf("------\n");
    }
    printf("\n");
#endif // TEST_uf_avgDelta

#ifdef TEST_mu
    printf("------------TEST_mu\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        for(j=0; j<node1->ns; ++j)
        {
            printf("%f  ", node1->mu[i*node1->ns+j]);
        }
        printf("\n---\n");
    }
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        for(j=0; j<node2->ns; ++j)
        {
            printf("%f  ", node2->mu[i*node2->ns+j]);
        }
        printf("\n---\n");
    }
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        for(j=0; j<node3->ns; ++j)
        {
            printf("%f  ", node3->mu[i*node3->ns+j]);
        }
        printf("\n---\n");
    }
#endif
    printf("\n");
#endif // TEST_mu

#ifdef TEST_observation
    printf("------------TEST_observation\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        for(j=0; j<node1->ns; ++j)
        {
            printf("%f  ", node1->observation[i*node1->ns+j]);
        }
        printf("\n---\n");
    }
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        for(j=0; j<node2->ns; ++j)
        {
            printf("%f  ", node2->observation[i*node2->ns+j]);
        }
        printf("\n---\n");
    }
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        for(j=0; j<node3->ns; ++j)
        {
            printf("%f  ", node3->observation[i*node3->ns+j]);
        }
        printf("\n---\n");
    }
#endif
    printf("\n");
#endif // TEST_observation

#ifdef TEST_beliefMal
    printf("------------TEST_beliefMal\n");
    printf("------Node: %d,0,0\n", currLayer);
    for(i=0; i<node1->nb; ++i)
    {
        printf("%f  ", node1->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#ifndef TEST_layer0
    printf("------Node: %d,0,0\n", currLayer-1);
    for(i=0; i<node2->nb; ++i)
    {
        printf("%f  ", node2->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#endif
#ifndef TEST_layer7
    printf("------Node: %d,0,0\n", currLayer+1);
    for(i=0; i<node3->nb; ++i)
    {
        printf("%f  ", node3->beliefMal[i]);
        // node1->beliefMal
        // node1->beliefEuc
    }
    printf("\n");
#endif
    printf("\n");
#endif // TEST_beliefMal

#endif // TEST_add
#endif // TEST_2013_5_30

	return 0;
}

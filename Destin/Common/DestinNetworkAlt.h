/*
 * destinnetworkalt.h
 *
 *  Created on: Jun 23, 2012
 *      Author: ted
 */

#ifndef DESTINNETWORKALT_H_
#define DESTINNETWORKALT_H_

#include <stdexcept>
#include <vector>
#include <iostream>

#include "INetwork.h"

extern "C" {
#include "destin.h"
#include "node.h"
}

#include "DestinIterationFinishedCallback.h"
#include <exception>

enum SupportedImageWidths {
    W4 = 4,     //1
    W8 = 8,     //2
    W16 = 16,   //3
    W32 = 32,   //4
    W64 = 64,   //5
    W128 = 128, //6
    W256 = 256, //7
    W512 = 512  //8 layers needed
};
#define MAX_IMAGE_WIDTH 512

class DestinNetworkAlt: public INetwork {

private:
    Destin * destin;
    bool training;
    DestinIterationFinishedCallback * callback;

    float beta;   //variance learning rate
    float lambda; //previous belief damping, 0 = disable, 1.0 = keep the same
    float gamma;  //parents previous belief damping, 0 = disable, 1.0 = keep the same

    float * temperatures; //one temperature per layer. If temperature = #
                          //centroids then the belief distribution doesn't change.
                          //Higher temperature makes the belief more winner
                          //take all, lower temperature makes it more uniform.

    /**
     * initTemperatures
     * Input layer stays the same. For the top two layers temperature = centroids * 2
     * Because the input layer is generally already sparse and the top layers
     * are more uniform and need help to becoming more decisive.
     */
    void initTemperatures(int layers, uint * centroids){
        temperatures = new float[layers];

        for(int l = 0 ; l < layers ; l++){
            temperatures[l] = centroids[l];
        }

        if(layers >= 2){
            temperatures[layers - 1] *= 2;
        }
        if(layers >=3){
            temperatures[layers - 2] *= 2;
        }
    }

public:
    DestinNetworkAlt(SupportedImageWidths width, unsigned int layers,
            unsigned int centroid_counts [] ) :
            training(true),
            beta(.01),
            lambda(.1),
            gamma(.1)
            {

        uint input_dimensionality = 16;
        uint c, l;
        callback = NULL;
        initTemperatures(layers, centroid_counts);
        temperatures = new float[layers];
        bool doesBoltzman = true;
        //figure out how many layers are needed to support the given
        //image width.
        bool supported = false;
        for (c = 4, l = 1; c <= MAX_IMAGE_WIDTH ; c *= 2, l++) {
            if (c == width) {
                supported = true;
                break;
            }
        }
        if(!supported){
            throw std::logic_error("given image width is not supported.");
        }
        if (layers != l) {
            throw std::logic_error("Image width does not match the given number of layers.");
        }
        uint n_classes = 0;//doesn't look like its used
        float starv_coef = 0.001;

        uint num_movements = 0; //this class does not use movements
        bool isUniform = true; //wheter nodes in a layer share centroids
        destin = InitDestin(
                input_dimensionality,
                layers,
                centroid_counts,
                n_classes,
                beta,
                lambda,
                gamma,
                temperatures,
                starv_coef,
                num_movements,
                isUniform,
                doesBoltzman
         );
        ClearBeliefs(destin);

        isTraining(true);

    }

    virtual ~DestinNetworkAlt() {
        if(destin!=NULL){
            DestroyDestin(destin);
            destin = NULL;
        }
        delete [] temperatures;
        temperatures = NULL;
    }

    void doDestin( //run destin with the given input
            float * input_dev //pointer to input memory on device
            ) {
        FormulateBelief(destin, input_dev);

        if(this->callback != NULL){
            this->callback->callback(*this );
        }
    }

    void isTraining(bool isTraining) {
        this->training = isTraining;
        for(int l = 0 ; l < destin->nLayers ; l++){
           destin->layerMask[l] = isTraining ? 1 : 0;
        }
    }

    bool isTraining() {
        return training;
    }

    void free() {
        DestroyDestin(destin);
        destin = NULL;
    }

    void setIsPOSTraining(bool training) {
        isTraining(training);
    }

    void setIsPSSATraining(bool no_op) {
        //this network doesn't do pssa training
        std::cout << "DestinNetworkAlt setIsPSSATraining called. Currently a noop.\n";
    }

    void setIterationFinishedCallback(DestinIterationFinishedCallback * callback){
        this->callback = callback;
    }

    int getBeliefsPerNode(int layer){
        return destin->nb[layer];
    }

    float * getNodeBeliefs(int layer, int row, int col){
        return getNode(layer, row, col)->beliefEuc;
    }
    Node * getNode(int layer, int row, int col){
        Node * n = GetNodeFromDestin(destin, layer, row, col);
        if(n == NULL){
            throw std::logic_error("could't getNode");
        }
        return n;
    }

    Destin * getNetwork(){
        return destin;
    }
    void displayFeatures(int layer, int node_start, int nodes){
        DisplayLayerFeatures(destin, layer, node_start, nodes);
    }

    /** print given node's centroid's locations
     *  The Centroids consists of 3 main parts, the dimensions which cluster
     *  on the nodes input, the dimensions which cluster on its previous belief
     *  and the dimensions which cluster on the parents previous belief.
     */
    void printNodeCentroidPositions(int layer, int row, int col){
        Node * n = getNode(layer, row, col);
        for(int centroid  = 0 ; centroid < n->nb ; centroid++){
            printf("centroid %i: input ", centroid);
            int dimension;
            for(dimension = 0 ; dimension < n->ni ; dimension++){
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf(" self ");
            for(int end = dimension + n->nb ; dimension < end ; dimension++){
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf(" parent ");
            for(int end = dimension + n->np ; dimension < end ; dimension++){
                printf("%.5f ", n->mu[centroid*n->ns + dimension]);
            }
            printf("\n");
        }
    }
};

#endif /* DESTINNETWORKALT_H_ */

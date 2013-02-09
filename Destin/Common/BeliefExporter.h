#ifndef BELIEF_EXPORTER_H
#define BELIEF_EXPORTER_H

#include <vector>
#include "DestinNetworkAlt.h"
#include <stdexcept>


/** Returns a pointer to a subset of destin's beliefs
  */
class BeliefExporter {

    DestinNetworkAlt & destin;
    uint outputSize;
    uint skipSize;
    short * winnerTree;
    const int winningTreeSize;
    uint labelBucket;
    int bottomLayer;
    const int nLayers;

    short getTreeLabelForCentroid(int centroid, int layer){
        return layer * labelBucket + centroid;
    }

    int buildTree(Node * parent, int pos){
        winnerTree[pos] = getTreeLabelForCentroid(parent->winner, parent->layer);
        if(parent->layer > bottomLayer && parent->children != NULL){
            for(int i = 0 ; i < 4 ; i++){
                pos = buildTree(parent->children[i], ++pos);
                winnerTree[++pos] = -1;
            }
        }
        return pos;
    }

public:

    /** Constructor
      * @param network - destin network to export belielfs
      * @param bottom - layer to start including the beliefs for. See getOutputSize()
      */
    BeliefExporter(DestinNetworkAlt & network, uint bottom):
        destin(network), winnerTree(NULL), bottomLayer(bottom),
        nLayers(destin.getLayerCount()),
        winningTreeSize((destin.getNetwork()->nNodes - 1) * 2 + 1)
    {
        labelBucket = ( 1 << ( sizeof(short) * 8 - 1))/nLayers;
        setBottomLayer(bottom);
        for(int i = 0 ; i <  nLayers; i++){
            if( network.getBeliefsPerNode(i) >= labelBucket){
                throw std::domain_error("BeliefExporter: too many centroids\n.");
            }
        }
    }

    ~BeliefExporter(){
        if(winnerTree!=NULL){
            delete [] winnerTree;
            winnerTree = NULL;
        }
    }

    void setBottomLayer(unsigned int bottom){
        if(bottom >= destin.getNetwork()->nLayers){
            throw std::domain_error("setBottomLayer: cannot be set to above the top layer\n");
        }

        Destin * d = destin.getNetwork();

        uint to_substract = 0;
        for(int i = 0 ; i < bottom ; i++){
            to_substract += d->layerSize[i] * d->nb[i];
        }
        outputSize = d->nBeliefs - to_substract;
        skipSize = to_substract;
    }

    /** Calulate how large the belief vector should be.
      * Starts a the top laver of the network and
      * adds up the space for each layer
      * stopping at and including the layer specified
      * by setBottomLayer()
      */
    unsigned int getOutputSize(){
        return outputSize;
    }

    /** Get the pointer to the begining of the destin beliefs.
      * The begining of the belief vector depends on what
      * the bottom layer is set to via the constructor or by
      * setBottomLayer methobottomLayerd. The end of the vector is
      * beliefs of the top layer node.
      * Do not delete or free this pointer.
      */
    float * getBeliefs(){
        return &(destin.getNetwork()->belief[skipSize]);
    }

    short * getWinningCentroidTree(){
        if(!destin.getNetwork()->isUniform){
            throw std::logic_error("BeliefExpoerter::getWinningCentroidTree only uniform destin is supported.\n");
        }
        if(winnerTree==NULL){
            winnerTree = new short[getWinningCentroidTreeSize()];
        }

        buildTree(destin.getNode(nLayers - 1, 0, 0), 0);
        return winnerTree;
    }

    int getWinningCentroidTreeSize(){
        return winningTreeSize;
    }

};

#endif

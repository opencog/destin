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
    const int nLayers;
    int bottomLayer;

public:

    /** Constructor
      * @param network - destin network to export belielfs
      * @param bottom - layer to start including the beliefs for. See getOutputSize()
      */
    BeliefExporter(DestinNetworkAlt & network, uint bottom):
        destin(network),
        nLayers(destin.getLayerCount())
    {
        setBottomLayer(bottom);
    }

    void setBottomLayer(unsigned int bottom){
        if(bottom >= destin.getNetwork()->nLayers){
            throw std::domain_error("setBottomLayer: cannot be set to above the top layer\n");
        }

        bottomLayer = bottom;
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
      * setBottomLayer method. The end of the vector is
      * beliefs of the top layer node.
      * Do not delete or free this pointer.
      */
    float * getBeliefs(){
        return &(destin.getNetwork()->belief[skipSize]);
    }



};

#endif

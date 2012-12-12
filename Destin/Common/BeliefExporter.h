#ifndef BELIEF_EXPORTER_H
#define BELIEF_EXPORTER_H

#include <vector>
#include "DestinNetworkAlt.h"
#include <stdexcept>

class BeliefExporter {

    DestinNetworkAlt & destin;
    uint outputSize;
    uint skipSize;
public:
    const unsigned int nLayers;

    BeliefExporter(DestinNetworkAlt & network, uint bottom):
        destin(network),
        nLayers(destin.getLayerCount())
        {
        //layerMask(nLayers, true){ //layerMask vector has nLayers elements all set to true
        setBottomLayer(bottom);
    }

    ~BeliefExporter(){
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

    float * getBeliefs(){
        return &(destin.getNetwork()->belief[skipSize]);
    }

};

#endif

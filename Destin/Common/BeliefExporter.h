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
    const int nLayers;
    int bottomLayer;
    float * beliefs;

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

    ~BeliefExporter(){
        deleteBeliefs();
    }

    void setBottomLayer(unsigned int bottom){
        if(bottom >= destin.getNetwork()->nLayers){
            throw std::domain_error("setBottomLayer: cannot be set to above the top layer\n");
        }

        bottomLayer = bottom;
        Destin * d = destin.getNetwork();

        outputSize = 0;
        for(int layer = bottom ; layer < d->nLayers ; layer++){
            outputSize += d->layerSize[layer] * d->nb[layer];
        }

        deleteBeliefs();
        beliefs = new float[outputSize];
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
      */
    float * getBeliefs(){
        uint beliefsOffset = 0;

        Destin * d = destin.getNetwork();

        for (uint layer = bottomLayer; layer < nLayers ; ++layer){
            destin.getLayerBeliefs(layer, beliefs + beliefsOffset);
            beliefsOffset += d->layerSize[layer] * d->nb[layer];
        }

        return beliefs;
    }

    void openMatFile(string matfile){

    }

    void closeMatFile(){

    }

    /**
     * This appends the current beliefs to the Mat file.
     * First it writes the label ( i.e. cifar image class)
     * that was used to generate the current beliefs.
     * Then it writes the concatonated beliefs given by the getBeliefs() method.
     * The lenght of the belief vector is given by getOutputSize()
     *
     * @brief writeBeliefToDisk
     * @param label - used to identify what type of input image was given to Destin that
     * led to the current output beliefs.
     */
    void writeBeliefToMat(int label){

    }

protected:

    void deleteBeliefs(){
        if (beliefs != NULL){
            delete[] beliefs;
            beliefs = NULL;
        }
    }

};

#endif

#ifndef BELIEF_EXPORTER_H
#define BELIEF_EXPORTER_H

#include <vector>
#include "DestinNetworkAlt.h"
#include <stdexcept>
#include <stdio.h>
#include <string.h>

/** Returns a pointer to a subset of destin's beliefs
  */
class BeliefExporter {

    DestinNetworkAlt & destin;
    uint outputSize;
    const int nLayers;
    int bottomLayer;
    float * beliefs;

    bool fileIsOpen;
    FILE *filePtr;
    string fileName;
public:

    /** Constructor
      * @param network - destin network to export belielfs
      * @param bottom - layer to start including the beliefs for. See getOutputSize()
      * @param file_name - uses this file name for the writeBeliefToDisk method.
      */
    BeliefExporter(DestinNetworkAlt & network, uint bottom):
        destin(network),
        fileIsOpen(false),
        fileName(""),
        filePtr(NULL),
        nLayers(destin.getLayerCount()),
        beliefs(NULL)
    {
        setBottomLayer(bottom);
    }

    ~BeliefExporter(){
        deleteBeliefs();
        if(fileIsOpen){
            fclose(filePtr);
            filePtr = NULL;
            fileIsOpen = false;
        }
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
      * Starts at the top layer of the network and
      * adds up the space for each layer
      * stopping at and including the layer specified
      * by setBottomLayer()
      */
    unsigned int getOutputSize(){
        return outputSize;
    }

    /** Get the pointer to the beginning of the destin beliefs.
      * The beginning of the belief vector depends on what
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

    /**
     * This appends the current destin beliefs to the file specified in the constructor.
     *
     * The first call to this method will create the file. The file is closed by the
     * closeBeliefFile() method or by the destructor.
     *
     * The data is written to file as plain text with columns seperated by tabs, and
     * rows delimited with a new line '\n' character.
     *
     * Each call to this method will append a new row to the file.
     *
     * The first column is the label. Then it writes the concatonated beliefs
     * vector given by the getBeliefs() method.
     * The lenght of the belief vector is given by getOutputSize()
     *
     * @brief writeBeliefToDisk
     * @param label - used to identify what type of input image (class) was given to Destin that
     * led to the current output beliefs. Is written to the first column of the output beliefs
     * @param file_name - File to write to. If this file name changes between calls,
     * then the current file is closed and a new file is opened with the file name.
     */
    void writeBeliefToDisk(int label, string file_name){
        float *beliefs = getBeliefs();
        int i = 0;

        if(fileIsOpen && file_name != fileName){
            // trying to write to a new file so close the curret one.
            closeBeliefFile();
        }

        if(!fileIsOpen){
            filePtr = fopen(file_name.c_str(),"w");
            if(filePtr == NULL){
                std::cerr << "Could not open file for writing beliefs!" << std::endl;
                return;
            }
            fileIsOpen = true;
            fileName = file_name;
        }
 	
        fprintf(filePtr, "%i\t", label);

        uint size = getOutputSize();
        for (i = 0; i < size; i++){
            fprintf(filePtr, "%.9f\t", beliefs[i]);
        }
        fprintf(filePtr, "\n");
    }

    void closeBeliefFile(){
        if(fileIsOpen){
            fclose(filePtr);
            fileIsOpen = false;
            filePtr = NULL;
        }
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

#ifndef INETWORK_H
#define INETWORK_H

#include <stdio.h>

class DestinIterationFinishedCallback;

class INetwork {
public:

    virtual ~INetwork(){}
    virtual void doDestin( float * dInput)=0;
    virtual void setIterationFinishedCallback(DestinIterationFinishedCallback * callback)=0;
    virtual void destroy()=0;
    virtual void setIsPSSATraining(bool training)=0;
    virtual void setIsPOSTraining(bool training)=0;
    virtual int getBeliefsPerNode(int layer)=0;
    virtual float * getNodeBeliefs(int layer, int row, int col)=0;
    virtual unsigned int getLayerCount() = 0;

    /** Prints a horizonal bar chart of the beliefs of a node.
      */
    void printBeliefGraph(int layer, int row, int col){
        float * b = getNodeBeliefs(layer, row, col);
        int nb = getBeliefsPerNode(layer);
        int barCharWidth = 40;
        for(int i = 0 ; i < nb ; i++){
            printf("%i: %f: ",i,b[i]);
            int width =  barCharWidth * b[i];
            if(width > barCharWidth || width < 0){
                printf("invalid belief\n");
                return;
            }
            for(int j = 0 ; j < width ; j++ ){
                printf("X");
            }
            printf("\n");
        }
    }

};
#endif

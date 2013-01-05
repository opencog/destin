#include "destin.h"
#include "learn_strats.h"

float CLS_Fixed(Destin * d,  Node * n, uint layer, uint centroid){
    return d->fixedLearnRate;
}

float CLS_Decay(Destin * d, Node * n, uint layer, uint centroid){
    uint wincount;
    float learnRate;
    if(d->isUniform){
        wincount = d->uf_persistWinCounts[layer][centroid];
        learnRate = wincount == 0 ? 0.0 : 1.0 / (float)wincount; //TODO: test persist win counts over multiple calls to FormulateBeliefs
    }else{
        learnRate = 1 / (float) n->nCounts[n->winner];
    }
    return learnRate;
}

#include "destin.h"
#include "learn_strats.h"

// TODO: make this not a global variable
// TODO: save the fixed learning rate in SaveDestin
float CLS_Fixed_learn_rate = 0.25;
void CLS_Fixed_SetRate(float rate){
    CLS_Fixed_learn_rate = rate;
}

float CLS_Fixed(Destin * d,  Node * n, uint layer, uint centroid){
    return CLS_Fixed_learn_rate;
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

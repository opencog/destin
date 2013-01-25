#include <math.h>
#include <string.h>
#include "belief_transform.h"
#include "node.h"
#include "destin.h"
#include "cent_image_gen.h"

void SetBeliefTransform(Destin * d, BeliefTransformEnum bt){

    d->beliefTransform = bt;
    switch(bt){
        case DST_BT_BOLTZ:
            d->beliefTransformFunc = &DST_BT_Boltzmann;
            break;
        case DST_BT_P_NORM:
            d->beliefTransformFunc = &DST_BT_PNorm;
            break;
        case DST_BT_NONE:
            d->beliefTransformFunc = &DST_BT_None;
            break;
        case DST_BT_WTA:
            d->beliefTransformFunc = &DST_BT_WinnerTakeAll;
            break;
        default:
            fprintf(stderr, "Warning: Invalid BeliefTransformEnum value %i. Setting null.\n", bt);
            d->beliefTransformFunc = NULL;
            break;
    }
    return;
}


 BeliefTransformEnum BeliefTransform_S_to_E(char * string){
    if(strcmp(string,"boltz")==0){
        return DST_BT_BOLTZ;
    }else if(strcmp(string,"pnorm")==0){
        return DST_BT_P_NORM;
    }else if(strcmp(string,"none")==0){
        return DST_BT_NONE;
    }else if(strcmp(string,"wta")==0){
        return DST_BT_WTA;
    }else{
        fprintf(stderr, "Warning: Invalid belief transform string: %s, defaulting to None.\n", string);
        return DST_BT_NONE;
    }
}

void DST_BT_Boltzmann(Node* n){
    // Start the process of exaggerating the probability distribution using Boltzman's method
    float maxBoltzEuc = 0;
    float maxBoltzMal = 0;
    uint i;
    for( i=0; i < n->nb; i++ ){
        // If so, check to see if the current Euclidean Boltzman belief is greater than our current max
        if( n->beliefEuc[i] * n->temp > maxBoltzEuc )
            // If so, update our max Euclidean Boltzman value.
            maxBoltzEuc = n->beliefEuc[i] * n->temp;

        // Check to see if the current Mahalanobis Boltzman belief is greater than our current max
        if( n->beliefMal[i] * n->temp > maxBoltzMal )
            // If so, update our max Mahalanobis Boltzman value.
            maxBoltzMal = n->beliefMal[i] * n->temp;
    }

    // Prepare Euclidean and Mahalanobis belief value
    float boltzEuc = 0;
    float boltzMal = 0;

    // Loop through the centroids
    for( i=0; i < n->nb; i++ )
    {
        // Recalculate beliefs with inclusion of Bolzmanish stuff
        n->beliefEuc[i] = exp(n->temp * n->beliefEuc[i] - maxBoltzEuc);
        n->beliefMal[i] = exp(n->temp * n->beliefMal[i] - maxBoltzMal);

        // Add the current belief to the totals
        boltzEuc += n->beliefEuc[i];
        boltzMal += n->beliefMal[i];
    }

    // Loop through the centroids AGAIN
    for( i=0; i < n->nb; i++ )
    {
        // Normalize the beliefs
        n->beliefEuc[i] /= boltzEuc;
        n->beliefMal[i] /= boltzMal;
    }
    return;
}

void DST_BT_PNorm(struct Node* n){
    Cig_PowerNormalize(n->beliefEuc, n->beliefEuc, n->nb, n->temp);
    Cig_PowerNormalize(n->beliefMal, n->beliefMal, n->nb, n->temp);
    return;
}

void DST_BT_None(struct Node* n){
    return;
}


void DST_BT_WinnerTakeAll(struct Node* n){
    int i;
    for(i = 0 ; i < n->nb ; i++){
        n->beliefEuc[i] = 0.0;
        n->beliefMal[i] = 0.0;
    }
    n->beliefEuc[n->winner] = 1.0;
    n->beliefMal[n->winner] = 1.0;
}


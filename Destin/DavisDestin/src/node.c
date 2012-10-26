#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "destin.h"
#include "node.h"
#include "macros.h"

#define EPSILON     1e-8

// CPU implementation of GetObservation kernel
void GetObservation( Node *n, float *framePtr, uint nIdx )
{
    n = &n[nIdx];

    uint i, j;
    uint ni, nb, np, ns, nc;

    ni = n->ni;
    nb = n->nb;
    np = n->np;
    ns = n->ns;
    nc = n->nc;

    if( n->inputOffsets == NULL )
    {
        // normal input
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = n->input[i];
        }
    } else {
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = framePtr[n->inputOffsets[i]];
        }
    }

    // set these to uniform for now.
    // TODO: REMOVE THIS WHEN RECURRENCE IS ENABLE
    for( i=0; i < nb; i++ )
    {
        n->observation[i+ni] = 1 / (float) nb; //n->pBelief[i] * n->gamma;
    }

    for( i=0; i < np; i++ )
    {
        n->observation[i+ni+nb] = 1 / (float) np; // n->parent_pBelief[i] * n->lambda;
    }

    for( i=0; i < nc; i++ )
    {
        n->observation[i+ni+nb+np] = 0;
    }
}

// CPU implementation of CalculateDistances kernel
void CalculateDistances( Node *n, uint nIdx )
{
    float delta;
    float sumEuc, sumMal;

    n = &n[nIdx];

    uint i, j;
    uint ni = n->ni;
    uint nb = n->nb;
    uint ns = n->ns;
    uint nc = n->nc;

   // iterate over each belief
    for( i=0; i < n->nb; i++ )
    {
        sumEuc = 0;
        sumMal = 0;
        uint bRow = i*ns;

        // iterate over each state for belief
        for( j=0; j < ns-nc; j++ )
        {
            delta = n->observation[j] - n->mu[bRow+j];

            delta *= delta;
            delta *= n->starv[i];

            sumEuc += delta;
            sumMal += delta / n->sigma[bRow+j];
        }

        n->genObservation[i] = sumMal;

        sumEuc = sqrt(sumEuc);
        sumMal = sqrt(sumMal);

        n->beliefEuc[i] = ( sumEuc < EPSILON ) ? 1 : (1 / sumEuc);
        n->beliefMal[i] = ( sumMal < EPSILON ) ? 1 : (1 / sumMal);
    }
}

// CPU implementation of NormalizeBelief kernel
void NormalizeBeliefGetWinner( Node *n, uint nIdx )
{
    n = &n[nIdx];
    
    float normEuc = 0;
    float normMal = 0;

    float maxEucVal = n->beliefEuc[0];
    uint maxEucIdx = 0;
    
    float maxMalVal = n->beliefMal[0];
    uint maxMalIdx = 0;

    uint i;

    for( i=0; i < n->nb; i++ )
    {
        normEuc += n->beliefEuc[i];
        normMal += n->beliefMal[i];

        if( n->beliefEuc[i] > maxEucVal )
        {
            maxEucVal = n->beliefEuc[i];
            maxEucIdx = i;
        }
        if( n->beliefMal[i] > maxMalVal )
        {
            maxMalVal = n->beliefMal[i];
            maxMalIdx = i;
        }
    }
    
    float maxBoltzEuc = 0;
    float maxBoltzMal = 0;

    // normalize beliefs to sum to 1
    for( i=0; i < n->nb; i++ )
    {
        n->beliefEuc[i] = ( normEuc < EPSILON ) ? (1 / (float) n->nb) : (n->beliefEuc[i] / normEuc);
        n->beliefMal[i] = ( normMal < EPSILON ) ? (1 / (float) n->nb) : (n->beliefMal[i] / normMal);

        // get maximum temp to normalize boltz normalization
        if( n->beliefEuc[i] * n->temp > maxBoltzEuc )
            maxBoltzEuc = n->beliefEuc[i] * n->temp;
        if( n->beliefMal[i] * n->temp > maxBoltzMal )
            maxBoltzMal = n->beliefMal[i] * n->temp;

        //n->pBelief[i] = n->beliefMal[i];
    }

    // boltzmann
    float boltzEuc = 0;
    float boltzMal = 0;

    for( i=0; i < n->nb; i++ )
    {
        n->beliefEuc[i] = exp(n->temp * n->beliefEuc[i] - maxBoltzEuc);
        n->beliefMal[i] = exp(n->temp * n->beliefMal[i] - maxBoltzMal);

        boltzEuc += n->beliefEuc[i];
        boltzMal += n->beliefMal[i];
    }

    for( i=0; i < n->nb; i++ )
    {
        n->beliefEuc[i] /= boltzEuc;
        n->beliefMal[i] /= boltzMal;

        n->pBelief[i] = n->beliefEuc[i];
    }

    n->winner = maxEucIdx;
    n->genWinner = maxMalIdx;
}

// CPU implementation for UpdateWinner kernel
void UpdateWinner( Node *n, uint *label, uint nIdx )
{
    // whoa!  zero comments here -- my bad

    // grab the node we want to work with.  this is a carryover
    // from the kernel behavior -- kind of like n = &n[blockIdx.x].
    // we could just pass a pointer to the node we actually want.
    n = &n[nIdx];

    // just an iterator
    uint i;

    // gets the row offset for the mu/sigma matrices to update
    uint winnerOffset = n->winner*n->ns;

    // the difference between an element of the observation and
    // an element of the mu matrix
    float delta;

    // increment number of counts for winning centroid.
    n->nCounts[n->winner]++;

    // keeps track of squared error for a node.  added together
    // with the sq. err. for all the other nodes in the network,
    // it gives you a feel for when the network approaches
    // convergence.
    n->muSqDiff = 0;

    // this is the offset in the observation vector where
    // the class labels start.
    uint ncStart = n->ni + n->nb + n->np;

    for( i=0; i < n->ns; i++ )
    {
        // if we are less than ncStart, we are not looking at
        // class labels.
        if( i < ncStart )
        {
            delta = n->observation[i] - n->mu[winnerOffset+i];

        // otherwise, use the class label to move the last n_c
        // components of the winning centroid
        } else {
            delta = (float) label[i - ncStart] - n->mu[winnerOffset+i];
        }

        // calculate how much we move the centroid.  the 1 / nCounts
        // term can be switched out for a different learning rate
        // whether fixed or adaptive.
        float dTmp = (1 / (float) n->nCounts[n->winner]) * delta;

        // move the winning centroid
        n->mu[winnerOffset+i] += dTmp;

        // increment the sq. difference
        n->muSqDiff += dTmp * dTmp;

        // update the variance of the winning centroid
        n->sigma[winnerOffset+i] += n->beta * (delta*delta - n->sigma[winnerOffset+i]);
    }
}

void UpdateStarvation(Node *n, uint nIdx)
{
    n = &n[nIdx];
    int i;
    for( i=0; i < n->nb; i++ )
    {
//        n->starv[i] = n->starv[i] * (1 - n->starvCoeff) + n->starvCoeff * (i == n->winner);
        n->starv[i] *= 1 - n->starvCoeff;
    }
    n->starv[n->winner] = 1;
}

void Uniform_UpdateStarvation(Node *n, uint nIdx)
{

}


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

    // Length of input vector
    ni = n->ni;

    // Number of centroids
    nb = n->nb;

    // Number of centroids of the parent
    np = n->np;

    // Sum of ni + nb + np + nc
    ns = n->ns;

    // 'Context'
    nc = n->nc;

    // Check to see if bottom layer image input data is available
    if( n->inputOffsets == NULL )
    {
        // If not, use input from the child nodes
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = n->input[i];
        }
    } else {
        // If so, use input from the input image
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = framePtr[n->inputOffsets[i]];
        }
    }

    // set these to uniform for now.
    // TODO: REMOVE THIS WHEN RECURRENCE IS ENABLE
    for( i=0; i < nb; i++ )
    {
        // Ignore previous beliefs
        n->observation[i+ni] = 1 / (float) nb;

        // Include previous beliefs
        // n->observation[i+ni] = n->pBelief[i] * n->gamma;
    }

    for( i=0; i < np; i++ )
    {
        // Ignore parent's previous belief
        n->observation[i+ni+nb] = 1 / (float) np;

        // Include parent's previous belief
        // n->observation[i+ni+nb] = n->parent_pBelief[i] * n->lambda;
    }

    for( i=0; i < nc; i++ )
    {
        // Apply context
        n->observation[i+ni+nb+np] = 0;
    }
}

// CPU implementation of CalculateDistances kernel
void CalculateDistances( Node *n, uint nIdx )
{
    // delta = difference between the input vector and a centroid
    float delta;
    // sumEuc = the Euclidean distance between the input vector and a centroid
    // sumMal = the Mahalanobis distance between the input vector and a centroid, taking the tightness of the cluster into account
    float sumEuc, sumMal;

    // Get a node from the pointer to the list of nodes
    n = &n[nIdx];

    // i = counter for loop through centroids
    // j = counter for loop through
    uint i, j;

    // Get the length of the node's (children's) input vector
    uint ni = n->ni;
    // Get the number of centroids in the node
    uint nb = n->nb;
    // Get the total length of the input vector
    uint ns = n->ns;
    // Get the context (?)
    uint nc = n->nc;


    // Get the sigma array depending on whether uniform or non-uniform destin is being used
    float * sigma = n->d->isUniform ? n->d->uf_sigma[n->layer] : n->sigma;
    // Get the dynamic starvation factor array depending on whether uniform or non-uniform destin is being used
    float * starv = n->d->isUniform ? n->d->uf_starv[n->layer] : n->starv;

   // iterate over each belief
    for( i=0; i < n->nb; i++ )
    {
        // Reset distances for the centroid that will be processed in this loop
        sumEuc = 0;
        sumMal = 0;

        // mu contains the probabilities (or grayscales) of the centroids in this node
        // bRow = start index of the probabilities (or grayscales) of the current centroid
        uint bRow = i*ns;

        // iterate over each state for belief
        // Loop through the items in the vector, ignoring the context
        for( j=0; j < ns-nc; j++ )
        {
            // mu contains the probabilities (or grayscales) of the centroids in this node

            // Calculate the distance between the input (observation) and the centroid's current location
            delta = n->observation[j] - n->mu[bRow+j];

            // Continuation of distance calculation
            delta *= delta;

            // Reduce the distance by the starvation factor
            delta *= starv[i];

            // Add the resulting distance to our Euclidean distance sum for this centroid
            sumEuc += delta;

            // Retrieve the sigma from the sigma array based on the centroid data column and add the distance to the Mahalanobis sum
            sumMal += delta / sigma[bRow+j];
        }

        // Dead code
        n->genObservation[i] = sumMal;

        // Take the square root of the distance to finalize the distance calculation
        sumEuc = sqrt(sumEuc);
        sumMal = sqrt(sumMal);

        // Calculate intermediate belief in the current centroid based on the distance between the centroid and the input vector data
        n->beliefEuc[i] = ( sumEuc < EPSILON ) ? 1 : (1 / sumEuc);
        n->beliefMal[i] = ( sumMal < EPSILON ) ? 1 : (1 / sumMal);
    }
}

// CPU implementation of NormalizeBelief kernel
void NormalizeBeliefGetWinner( Node *n, uint nIdx )
{
    // Get a node from the pointer to the list of nodes
    n = &n[nIdx];
    
    // Define variables for normalized Euclidean and Mahalanobis distance
    float normEuc = 0;
    float normMal = 0;

    // Pick a value from the Euclidean beliefs to initialize the maxEucVal variable
    float maxEucVal = n->beliefEuc[0];
    uint maxEucIdx = 0;
    
    // Set the index of the current max Euclidean belief value to the index of the value we just retrieved
    float maxMalVal = n->beliefMal[0];
    uint maxMalIdx = 0;

    // Declare looping integer (C requirement)
    uint i;

    // Loop through the centroids in this node
    for( i=0; i < n->nb; i++ )
    {
        // Sum the beliefs to use for normalization later
        normEuc += n->beliefEuc[i];
        normMal += n->beliefMal[i];

        // Check to see if the current Euclidean belief is greater than our current max
        if( n->beliefEuc[i] > maxEucVal )
        {
            // If so, update our max Euclidean belief value and its index
            maxEucVal = n->beliefEuc[i];
            maxEucIdx = i;
        }
        // Check to see if the current Mahalanobis belief is greater than our current max
        if( n->beliefMal[i] > maxMalVal )
        {
            // If so, update our max Mahalanobis belief value and its index
            maxMalVal = n->beliefMal[i];
            maxMalIdx = i;
        }
    }
    
    // Start the process of exaggerating the probability distribution using Boltzman's method
    float maxBoltzEuc = 0;
    float maxBoltzMal = 0;

    // Check to see if we want to apply the Boltzman exaggeration method
    bool boltzman = n->d->doesBoltzman;


    // normalize beliefs to sum to 1
    for( i=0; i < n->nb; i++ )
    {
        // If the sum of all beliefs is lower than EPSILON, calculate the belief probability for the
        // current centroid as a constant part of the number of centroids. Else calculate it as a part of
        // the sum of our beliefs.
        n->beliefEuc[i] = ( normEuc < EPSILON ) ? (1 / (float) n->nb) : (n->beliefEuc[i] / normEuc);
        n->beliefMal[i] = ( normMal < EPSILON ) ? (1 / (float) n->nb) : (n->beliefMal[i] / normMal);

        // Check to see whether Boltzman should be applied
        if(boltzman){
            // If so, check to see if the current Euclidean Boltzman belief is greater than our current max
            if( n->beliefEuc[i] * n->temp > maxBoltzEuc )
                // If so, update our max Euclidean Boltzman value.
                maxBoltzEuc = n->beliefEuc[i] * n->temp;

            // Check to see if the current Mahalanobis Boltzman belief is greater than our current max
            if( n->beliefMal[i] * n->temp > maxBoltzMal )
                // If so, update our max Mahalanobis Boltzman value.
                maxBoltzMal = n->beliefMal[i] * n->temp;
        }else{
            // Else use the non exaggerated belief value
            n->pBelief[i] = n->beliefMal[i];
        }
    }
    // Check to see whether to apply Boltzman
    if(boltzman){
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

            // Set the belief to be used in the end
            n->pBelief[i] = n->beliefEuc[i];
        }
    }

    // Set the winning centroid of the current node to the index of the centroid with the highest Euclidean belief value
    n->winner = maxEucIdx;

    //TODO: test that this works for non uniform
    if(n->d->isUniform){

        // Add one to the current iteration's wincount of the winning centroid of the node's layer (since this centroid can be
        // shared between multiple nodes in the same layer)
        long c = ++(n->d->uf_winCounts[n->layer][n->winner]); //used when averaging the delta vectors

        // For the first node that declares this centroid the winner update the persistent array of win counts.
        if( c == 1){//only increment this once even if multiple nodes pick this shared centroid
            n->d->uf_persistWinCounts[n->layer][n->winner]++;
        }
    }

    // Todo: write useful comment here
    n->genWinner = maxMalIdx;
}


void CalcCentroidMovement( Node *n, uint *label, uint nIdx )
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
        n->delta[i] = delta;
    }
    return;
}


float CLS_Fixed(Destin * d,  Node * n, uint layer, uint centroid){
    float learningRate = 0.25;
    return learningRate;
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

//average the deltas and multiply them by the shared ncounts
void Uniform_AverageDeltas(Node * n, uint nIdx){
    n = &n[nIdx];
    uint s;

    int count;
    for(s = 0; s < n->ns ; s++){
        count = n->d->uf_winCounts[n->layer][n->winner];
        if(count > 0){
            n->d->uf_avgDelta[n->layer][n->winner * n->ns + s] += n->delta[s] / (float)count;
        }
    }
    return;
}

void Uniform_ApplyDeltas(Destin * d, uint layer, float * layerSharedSigma){
    uint c, s, ns;
    float diff, learnRate, dt;
    for(c = 0 ; c < d->nb[layer]; c++){
        learnRate = d->centLearnStratFunc(d, NULL, layer, c);
        Node * n = GetNodeFromDestin(d, layer, 0,0);
        ns = n->ns;
        for(s = 0 ; s < ns ; s++){
            //move the centroid with the averaged delta
            dt = d->uf_avgDelta[layer][c * ns + s];
            diff = dt * learnRate;
            n->mu[c * ns + s] += diff; //all nodes in a layer share this n->mu pointer
            n->muSqDiff += diff * diff; //only 0th node of each layer gets a muSqDiff
            //TODO: write unit test for layerSharedSigma
            layerSharedSigma[c * ns + s] += n->beta * (dt * dt - layerSharedSigma[c * ns + s]  );
        }
    }
    return;
}


void MoveCentroids( Node *n, uint nIdx ){
    n = &n[nIdx];
    
    // gets the row offset for the mu/sigma matrices to update
    uint winnerOffset = n->winner*n->ns;

    // increment number of counts for winning centroid.
    n->nCounts[n->winner]++;

    int i;
    float learnRate = n->d->centLearnStratFunc(n->d, n, 0, 0);
    for(i = 0 ; i < n->ns ; i++){   
        // calculate how much we move the centroid.  the 1 / nCounts
        // term can be switched out for a different learning rate
        // whether fixed or adaptive.
        float delta = n->delta[i];
        float dTmp = learnRate * delta;

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

void Uniform_UpdateStarvation(Destin * d, uint layer, float * sharedStarvation, uint * sharedCentroidsWinCounts, float starvCoeff)
{
    uint i, nb = d->nb[layer];
    for(i = 0; i < nb ; i++){
        sharedStarvation[i] *= 1 - starvCoeff;
        if(sharedCentroidsWinCounts[i] > 0){
            sharedStarvation[i] = 1;
        }
    }

}


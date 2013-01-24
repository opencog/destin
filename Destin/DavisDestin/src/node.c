#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "destin.h"
#include "node.h"
#include "macros.h"

#define EPSILON     1e-8
#define MAX_INTERMEDIATE_BELIEF (1.0 / EPSILON)

//#define RECURRENCE_ON

#define USE_MAL
//#define USE_EUC

#if defined(USE_MAL) && defined(USE_EUC)
#error "both USE_MAL and USE_EUC can't be set"
#endif

//#define STARV_QUICK_RESET
#define STARV_SLOW_RESET


#ifdef UNIT_TEST
    //checks various things if this is compiled for the unit test
    #define CHECK_NORM_LESS_THAN_EPSILON
    #define CHECK_BELIEF_ZERO
    #define CHECK_OBS
    #define CHECK_BIG_MU
#define checkinf(d){\
    if(isinf(d)){\
    oops("was inf at line %i\n", __LINE__ );\
    }\
}
#endif

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
    if( n->layer > 0 )
    {
        // If not, use input from the child nodes
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = n->input[n->inputOffsets[i]];
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
#ifdef RECURRENCE_ON
        n->observation[i+ni] = n->pBelief[i] * n->gamma;
#else
        n->observation[i+ni] = 1 / (float) nb;
#endif
    }

    for( i=0; i < np; i++ )
    {
#ifdef RECURRENCE_ON
        n->observation[i+ni+nb] = n->parent_pBelief[i] * n->lambda;
#else
        n->observation[i+ni+nb] = 1 / (float) np;
#endif
    }

    for( i=0; i < nc; i++ )
    {
        // Apply context
        n->observation[i+ni+nb+np] = 0;
    }

#ifdef CHECK_OBS
    for(i = 0 ; i < n->ns ; i++){
        float o = n->observation[i];
        if(isinf(o)){
            oops("observation was inf at index %i\n", i);
        }
        if(isnan(o)){
            oops("observation was nan at index %i\n", i);
        }
        if(o < 0){
            oops("observation was negative at index %i\n", i);
        }
        if(o > 1){
            oops("observation was greater than 1.0 at index %i\n", i);
        }
    }
#endif
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

    // Get the total length of the input vector
    const uint ns = n->ns;
    // Get the context (?)
    const uint nc = n->nc;


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

            // Calculate the difference between the input (observation) and the centroid's current location
            delta = n->observation[j] - n->mu[bRow+j];
            // Start distance calculation
            delta *= delta;
#ifdef CHECK_BELIEF_ZERO
            if(isnan(delta)){
                oops("delta was nan\n");
            }
            if(isinf(delta)){
                oops("delta was inf. obs: %e, mu: %e\n", n->observation[j], n->mu[bRow + j]);
            }
#endif
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
        n->beliefEuc[i] = ( sumEuc < EPSILON ) ? MAX_INTERMEDIATE_BELIEF : (1.0 / sumEuc);
        n->beliefMal[i] = ( sumMal < EPSILON ) ? MAX_INTERMEDIATE_BELIEF : (1.0 / sumMal);

#ifdef CHECK_BELIEF_ZERO
        if(n->beliefEuc[i] < EPSILON){
            oops("n->beliefEuc == 0, sumEuc:%e \n", sumEuc);
        }
        if(n->beliefMal[i] < EPSILON){
            oops("n->beliefMal == 0, sumMal:%e\n", sumMal);
        }
#endif
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
    
#ifdef CHECK_NORM_LESS_THAN_EPSILON
    if (normEuc < EPSILON){
        oops("oops: normEuc was less than EPSILON: %e\n", normEuc);
    }
    if (normMal < EPSILON){
        oops("oops: normMal was less than EPSILON: %e\n", normMal);
    }
#endif


    // normalize beliefs to sum to 1
    for( i=0; i < n->nb; i++ )
    {
        n->beliefEuc[i] = n->beliefEuc[i] / normEuc;
        n->beliefMal[i] = n->beliefMal[i] / normMal;
    }

    n->d->beliefTransformFunc(n);

    for( i=0; i < n->nb; i++ ){
#ifdef USE_MAL
        n->pBelief[i] = n->beliefMal[i];
#endif
#ifdef USE_EUC
        n->pBelief[i] = n->beliefEuc[i];
#endif
    }

    // Set the winning centroid of the current node to the index of the centroid with the highest Euclidean belief value
    n->winner = maxEucIdx;

    //TODO: test that this works for non uniform
    if(n->d->isUniform){

        // Add one to the current iteration's wincount of the winning centroid of the node's layer (since this centroid can be
        // shared between multiple nodes in the same layer)
        long c;
        #pragma omp critical
        {
            c = ++(n->d->uf_winCounts[n->layer][n->winner]); //used when averaging the delta vectors
        }//omp critical

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
    uint winnerOffset = n->winner * n->ns;

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

//average the deltas and multiply them by the shared ncounts
void Uniform_AverageDeltas(Node * n, uint nIdx){
    n = &n[nIdx];
   int count = n->d->uf_winCounts[n->layer][n->winner];
   if(count > 0){
        uint s;
        for(s = 0; s < n->ns ; s++){
            n->d->uf_avgDelta[n->layer][n->winner * n->ns + s] += n->delta[s] / (float)count;
        }
    }
    return;
}

void Uniform_ApplyDeltas(Destin * d, uint layer, float * layerSharedSigma){
    uint c, s, ns;
    float diff, learnRate, dt;

    //iterate over shared centroids
    Node * n = GetNodeFromDestin(d, layer, 0,0);
    for(c = 0 ; c < d->nb[layer]; c++){

        //use learning strategy function pointer to get the learning rate
        learnRate = d->centLearnStratFunc(d, NULL, layer, c);

        //get the first node of the current layers
        ns = n->ns;
        for(s = 0 ; s < ns ; s++){
            //move the centroid with the averaged delta
            dt = d->uf_avgDelta[layer][c * ns + s];
            diff = dt * learnRate;
            n->mu[c * ns + s] += diff; //all nodes in a layer share this n->mu pointer
#ifdef CHECK_BIG_MU
            if ( n->mu[c * ns + s] > 1.0){
                oops("Big mu value:%e at line %i\n",n->mu[c * ns + s],__LINE__ );
            }
#endif
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
#ifdef CHECK_BIG_MU
        if (n->mu[winnerOffset+i]  > 1.0){
            oops("Big mu value:%e at line %i\n",n->mu[winnerOffset+i] ,__LINE__ );
        }
#endif
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
#ifdef STARV_QUICK_RESET
    for( i=0; i < n->nb; i++ )
    {
        n->starv[i] *= 1 - n->starvCoeff;
    }
    n->starv[n->winner] = 1;
#endif

#ifdef STARV_SLOW_RESET
    for( i=0; i < n->nb; i++ )
    {
         n->starv[i] = n->starv[i] * (1 - n->starvCoeff) + n->starvCoeff * (i == n->winner);
    }
#endif
}

void Uniform_UpdateStarvation(Destin * d, uint layer, float * sharedStarvation, uint * sharedCentroidsWinCounts, float starvCoeff)
{
    uint i, nb = d->nb[layer];
#ifdef STARV_QUICK_RESET
    for(i = 0; i < nb ; i++){
        sharedStarvation[i] *= 1 - starvCoeff;
        if(sharedCentroidsWinCounts[i] > 0){
            sharedStarvation[i] = 1;
        }
    }
#endif
#ifdef STARV_SLOW_RESET
    for(i = 0; i < nb ; i++){
        sharedStarvation[i] = sharedStarvation[i] * (1 - starvCoeff) + starvCoeff * ( sharedCentroidsWinCounts[i] > 0 );
    }
#endif

}


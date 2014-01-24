#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "destin.h"
#include "node.h"
#include "macros.h"
#include "centroid.h"

//checks various things if this is compiled for the unit test
#ifdef UNIT_TEST
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

void GetObservation( Node *n, float *framePtr, uint nIdx )
{
    n = &n[nIdx];
    bool isRecurrent = n->d->isRecurrent;
    uint i, j, k;
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
        i = 0;
        for ( j = 0; j < n->nChildren; j++ )
        {
            // children may be NULL i.e. if childsNumber is not square for DeSTIN square geometry
            if ( n->children[j] != NULL )
            {
                // copy child's output
                for ( k = 0; k < n->children[j]->nb; k++, i++ )
                {
                    n->observation[i] = n->children[j]->outputBelief[k];
                }
            } else {
                // child not exists, fill in zeros
                // Number of beliefs for child nodes
                uint childBeliefs = n->d->nb[n->layer - 1];
                for ( k = 0; k < childBeliefs; k++, i++ )
                {
                    n->observation[i] = 0;
                }
            }
        }
    } else {
        // If so, use input from the input image
        for( i=0; i < ni; i++ )
        {
            n->observation[i] = framePtr[n->inputOffsets[i]];
        }
    }

    //If recurrent, get observation from node's previous belief and parent's previous belief.
    if(isRecurrent){
        for( i=0; i < nb; i++ ){
            // get previous beliefs
            n->observation[i+ni] = n->outputBelief[i] * n->gamma;
        }
        for( i=0; i < np; i++ ){

            // if have a parent
            if (n->firstParent != NULL){
                // get parent's beliefs
                n->observation[i+ni+nb] = n->firstParent->outputBelief[i] * n->lambdaCoeff;
            } else { // does not have a parent, must be the top node
                n->observation[i+ni+nb] = 0;
            }
        }
    }else{ // No recurrence so set these to uniform for now.
        for( i=0; i < nb; i++ ){
            n->observation[i+ni] = 1 / (float) nb;
        }
        for( i=0; i < np; i++ ){
            n->observation[i+ni+nb] = 1 / (float) np;
        }
    }

    for( i=0; i < nc; i++ )
    {
        // Apply context
        n->observation[i+ni+nb+np] = 0;
    }

    // Apply extended input
    // Adds an extra length to the observation of length inputImageSize * ( extRatio - 1)
    if(n->layer == 0)
    {
        for(j=1; j<n->d->extRatio; ++j)
        {
            for(i=0; i<ni; ++i)
            {
                n->observation[i+j*ni+nb+np+nc] = framePtr[n->inputOffsets[i] + n->d->inputImageSize*j];
            }
        }
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

    // Get the total length of the input vector
    const uint ns = n->ns;
    // Get the context (?)
    const uint nc = n->nc;


    // Get the sigma array depending on whether uniform or non-uniform destin is being used
    float ** sigma = n->d->isUniform ? n->d->uf_sigma[n->layer] : n->sigma;
    // Get the dynamic starvation factor array depending on whether uniform or non-uniform destin is being used
    float * starv = n->d->isUniform ? n->d->uf_starv[n->layer] : n->starv;

    // iterate over each belief
    for( i=0; i < n->nb; i++ )
    {
        // Reset distances for the centroid that will be processed in this loop
        sumEuc = 0;
        sumMal = 0;

        // iterate over each state for belief
        // Loop through the items in the vector, ignoring the context
        for( j=0; j < ns-nc; j++ )
        {
            // mu contains the probabilities (or grayscales) of the centroids in this node

            // Calculate the difference between the input (observation) and the centroid's current location
            delta = n->observation[j] - n->mu[i][j];
            // Start distance calculation
            delta *= delta;

#ifdef CHECK_BELIEF_ZERO
            if(isnan(delta)){
                oops("delta was nan\n");
            }
            if(isinf(delta)){
                oops("delta was inf. obs: %e, mu: %e\n", n->observation[j], n->mu[i][j]);
            }
#endif
            // Reduce the distance by the starvation factor
            delta *= starv[i];

            // Add the resulting distance to our Euclidean distance sum for this centroid
            sumEuc += delta;

            // Retrieve the sigma from the sigma array based on the centroid data column and add the distance to the Mahalanobis sum
            sumMal += delta / sigma[i][j];
        }

        // Take the square root of the distance to finalize the distance calculation
        sumEuc = sqrt(sumEuc);
        sumMal = sqrt(sumMal);

        // Calculate intermediate belief in the current centroid based on the distance between the centroid and the input vector data
        /*n->beliefEuc[i] = ( sumEuc < EPSILON ) ? MAX_INTERMEDIATE_BELIEF : (1.0 / sumEuc);
        n->beliefMal[i] = ( sumMal < EPSILON ) ? MAX_INTERMEDIATE_BELIEF : (1.0 / sumMal);*/

        // 2013.9.4
        // CZT: a new method for distance-to-probability
        n->beliefEuc[i] = 1 / (1+sumEuc);
        n->beliefMal[i] = 1 / (1+sumMal);

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
    // Set the winning centroid of the current node to the index of the centroid with the highest Euclidean belief value
#ifdef USE_MAL
    n->winner = maxMalIdx;
#endif
#ifdef USE_EUC
    n->winner = maxEucIdx;
#endif

#ifdef CHECK_NORM_LESS_THAN_EPSILON
    if (normEuc < EPSILON){
        oops("oops: normEuc was less than EPSILON: %e\n", normEuc);
    }
    if (normMal < EPSILON){
        oops("oops: normMal was less than EPSILON: %e\n", normMal);
    }
#endif


    // normalize beliefs to sum to 1
    // TODO: think beliefTranform already does this
    for( i=0; i < n->nb; i++ )
    {
        n->beliefEuc[i] = n->beliefEuc[i] / normEuc;
        n->beliefMal[i] = n->beliefMal[i] / normMal;
    }

    n->d->beliefTransformFunc(n);

    for( i=0; i < n->nb; i++ ){
#ifdef USE_MAL
        n->belief[i] = n->beliefMal[i];
#endif
#ifdef USE_EUC
        n->belief[i] = n->beliefEuc[i];
#endif
    }


    //TODO: test that this works for non uniform
    if(n->d->isUniform){

        // Add one to the current iteration's wincount of the winning centroid of the node's layer (since this centroid can be
        // shared between multiple nodes in the same layer)
        long c;
        #pragma omp critical //open mp, only 1 thread in this section
        {
            c = ++(n->d->uf_winCounts[n->layer][n->winner]); //used when averaging the delta vectors
        }//omp critical

        // For the first node that declares this centroid the winner update the persistent array of win counts.
        if( c == 1){//only increment this once even if multiple nodes pick this shared centroid
            n->d->uf_persistWinCounts[n->layer][n->winner]++;
        }
        // 2013.7.3
        // CZT: the detailed record of winning;
        if(n->d->uf_persistWinCounts_detailed != NULL)
        {
            n->d->uf_persistWinCounts_detailed[n->layer][n->winner]++;
        }
    }

    // Todo: write useful comment here
    n->genWinner = maxMalIdx;
}

void CalcCentroidMovement( Node *n, uint *label, uint nIdx )
{
    // grab the node we want to work with.  this is a carryover
    // from the kernel behavior -- kind of like n = &n[blockIdx.x].
    // we could just pass a pointer to the node we actually want.
    n = &n[nIdx];

    // just an iterator
    uint i;

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
    uint ncEnd = n->ni + n->nb + n->np + n->nc;

    // 2013.7.3
    // CZT: to keep the original structure, I still add the if-statement to
    // check nc;
    for( i=0; i < n->ns; i++ )
    {
        if(i>=ncStart && i<ncEnd)
        {
            delta = (float) label[i - ncStart] - n->mu[n->winner][i];
        }
        else
        {
            delta = n->observation[i] - n->mu[n->winner][i];
        }
        n->delta[i] = delta;
    }
    return;
}

//average the deltas and multiply them by the shared ncounts
void Uniform_AverageDeltas(Node * n, uint nIdx){
    n = &n[nIdx];
    int count = n->d->uf_winCounts[n->layer][n->winner];
    uint s;
    if(count > 0){
        for(s = 0; s < n->ns ; s++){
            n->d->uf_avgDelta[n->layer][n->winner][s] += n->delta[s] / (float)count;
            n->d->uf_avgSquaredDelta[n->layer][n->winner][s] += n->delta[s]*n->delta[s] / (float)(count);
        }
    }

    for(s = 0; s < n->ns ; s++){
        n->d->uf_avgAbsDelta[n->layer][s] += fabs(n->delta[s]) / (float)n->d->layerSize[n->layer];
    }
}

// CZT: uniform;
void Uniform_ApplyDeltas(Destin * d, uint layer, float ** layerSharedSigma){
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
            dt = d->uf_avgDelta[layer][c][s];
            diff = dt * learnRate;
            n->mu[c][s] += diff; //all nodes in a layer share this n->mu pointer
#ifdef CHECK_BIG_MU
            if ( n->mu[c][s] > 1.0){
                oops("Big mu value:%e at line %i\n",n->mu[c][s],__LINE__ );
            }
#endif
            n->muSqDiff += diff * diff; //only 0th node of each layer gets a muSqDiff
            //TODO: write unit test for layerSharedSigma
            //layerSharedSigma[c * ns + s] += n->beta * (dt * dt - layerSharedSigma[c * ns + s]);
            layerSharedSigma[c][s] += n->beta * (d->uf_avgSquaredDelta[layer][c][s] - layerSharedSigma[c][s]);
        }
    }
    for(s = 0 ; s < ns ; s++){
        d->uf_absvar[layer][s] += n->beta * (d->uf_avgAbsDelta[layer][s] - d->uf_absvar[layer][s]);
    }
    return;
}

// CZT: for non-uniform;
void MoveCentroids( Node *n, uint nIdx ){
    n = &n[nIdx];
    
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
        n->mu[n->winner][i] += dTmp;
#ifdef CHECK_BIG_MU
        if (n->mu[n->winner][i]  > 1.0){
            oops("Big mu value:%e at line %i\n",n->mu[n->winner][i] ,__LINE__ );
        }
#endif
        // increment the sq. difference
        n->muSqDiff += dTmp * dTmp;

        // update the variance of the winning centroid
        n->sigma[n->winner][i] += n->beta * (delta*delta - n->sigma[n->winner][i]);

        // Ben:
        //n->absvar[winnerOffset+i]  += n->beta * ( fabs(delta) - n->absvar[winnerOffset+i]);
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

void Uniform_UpdateFrequency(Destin * d, uint layer, float * sharedFrequency, uint * sharedCentroidsWinCounts, float freqCoeff)
{
    uint i, nb = d->nb[layer];

    float coeff = freqCoeff / (1 + layer);
    for(i = 0; i < nb ; i++){
        sharedFrequency[i] = sharedFrequency[i] * (1 - coeff) +
                             coeff * ( sharedCentroidsWinCounts[i] / (float) d->layerSize[layer]);
    }
}

//TODO: optimize to delete many centroids at one shot
void Uniform_DeleteCentroids(Destin *d)
{
    uint nb;
    int l, c, i;

    for (l = 0; l < d->nLayers; l++)
    {
        // layer not trained
        if (d->layerMask[l] == 0)
        {
            continue;
        }
        nb = d->nb[l];
        for (c = nb - 1; c >= 0; c--)
        {
            if (d->uf_winFreqs[l][c] < 1/(float) nb * d->freqTreshold)
            {
                DeleteUniformCentroid(d, l, c);
            }
        }
    }
}

void Uniform_AddNewCentroids(Destin *d)
{
    uint l, i;
    float absvar;

    for (l = 0; l < d->nLayers; l++)
    {
        // layer not trained or belief dimensionality reached
        if (d->layerMask[l] == 0 || d->nb[l] >= d->layerMaxNb[l])
        {
            continue;
        }

        Node * n = GetNodeFromDestin(d, l, 0, 0);
        absvar = 0;
        for (i = 0; i < n->ns; i++)
        {
            absvar += d->uf_absvar[l][i];
        }
        absvar /= n->ns;
        float rnd = (float) rand() / (float) RAND_MAX;
        if (rnd < d->addCoeff * absvar)
        {
            AddUniformCentroid(d, l);
        }
    }
}

void UpdateNodeSizes(Node * n, uint ni, uint nb, uint np, uint nc)
{
    n->ni = ni;
    n->nb = nb;
    n->np = np;
    n->nc = nc;

    uint ns = nb + np + nc + ((n->layer == 0) ? ni*n->d->extRatio : ni);
    n->ns = ns;

    if (nb > n->d->maxNb)
    {
        n->d->maxNb = nb;
    }
    if (ns > n->d->maxNs)
    {
        n->d->maxNs == ns;
    }
}

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "macros.h"
#include "node.h"
#include "destin.h"


// create a destin instantiation from a file
Destin * CreateDestin( char *filename ) {
    Destin *newDestin;

    FILE *configFile;
    // open file
    configFile = fopen(filename, "r");
    if( !configFile ) {
        fprintf(stderr, "Cannot open config file %s\n", filename);
        exit(1);
    }
    
    uint ni, nl, nc, i, nMovements;
    uint *nb;
    float beta, lambda, gamma, starvCoeff;
    float *temp;

    // parse config file

    // get number of movements per digit
    fscanf(configFile, "%d", &nMovements);
    printf("nMovements: %d\n", nMovements);

    // get number of distinct classes
    fscanf(configFile, "%d", &nc);
    printf("# classes: %d\n", nc);

    // get input dimensionality
    fscanf(configFile, "%d", &ni);
    printf("input dim: %d\n", ni);

    // get number of layers
    fscanf(configFile, "%d", &nl);
    nb = (uint *) malloc(sizeof(uint) * nl);
    temp = (float *) malloc(sizeof(float) * nl);

    printf("# layers: %d\n", nl);

    // get layer beliefs and temps
    for(i=0; i < nl; i++) {
        fscanf(configFile, "%d %f", &nb[i], &temp[i]);
        printf("\t%d %f\n", nb[i], temp[i]);
    }

    // get coeffs
    fscanf(configFile, "%f", &beta);
    fscanf(configFile, "%f", &lambda);
    fscanf(configFile, "%f", &gamma);
    fscanf(configFile, "%f", &starvCoeff);
    printf("beta: %0.2f. lambda: %0.2f. gamma: %0.2f. starvCoeff: %0.2f\n", beta, lambda, gamma, starvCoeff);

    bool isUniform = false;
    newDestin = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff, nMovements, isUniform);

    fclose(configFile);

    free(nb);
    free(temp);

    return newDestin;
}

Destin * InitDestin( uint ni, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma, float *temp, float starvCoeff, uint nMovements, bool isUniform )
{
    uint nNodes, nInputPipeline;
    uint i, l, nBeliefs, maxNb, maxNs;
    size_t bOffset, iOffset;


    Destin *d;

    // initialize a new Destin object
    MALLOC(d, Destin, 1);

    d->nNodes = 0;
    d->nLayers = nl;

    d->nMovements = nMovements;
    d->isUniform = isUniform;
    MALLOC(d->inputLabel, uint, nc);
    for( i=0; i < nc; i++ )
    {
        d->inputLabel[i] = 0;
    }
    d->nc = nc;

    MALLOC(d->temp, float, nl);
    memcpy(d->temp, temp, sizeof(float)*nl);

    MALLOC(d->nb, uint, nl);
    memcpy(d->nb, nb, sizeof(uint)*nl);

    // get number of nodes to allocate
    // starting from the top layer with one node,
    // each subsequent layer has 4x the nodes.
    //
    // eg., 2-layer: 05 nodes
    //      3-layer: 21 nodes
    //      4-layer: 85 nodes
    //
    // also keep track of the number of beliefs
    // to allocate

    MALLOC(d->layerSize, uint, nl);

    nNodes = 0;
    nBeliefs = 0;
    for( i=0, l=nl ; l != 0; l--, i++ )
    {
        d->layerSize[i] = 1 << 2*(l-1);

        nNodes += d->layerSize[i];
        nBeliefs += d->layerSize[i] * nb[i];
    }

    d->nNodes = nNodes;

    // input pipeline -- all beliefs are copied from the output of each
    // node to the input of the next node on each timestep. we want
    // the belief of each node except for the top node (its output goes
    // to no input to another node) to be easily copied to the input
    // of the next node, so we allocate a static buffer for it.
    nInputPipeline = nBeliefs - nb[nl-1];

    d->nInputPipeline = nInputPipeline;

    // allocate node pointers on host
    MALLOC(d->nodes, Node, nNodes);

    // allocate space for inputs to nodes
    MALLOC(d->inputPipeline, float, nInputPipeline);

    // allocate space for beliefs for nodes on host
    MALLOC(d->belief, float, nBeliefs);

    d->nBeliefs = nBeliefs;

    // allocate for each layer an array of size number = n centroids for that layer
    // that counts how many nodes in a layer pick the given centroid as winner.
    MALLOC(d->sharedCentroidsWinCounts, uint *, d->nLayers);
    for(l = 0 ; l < d->nLayers ; l++){
        MALLOC( d->sharedCentroidsWinCounts[l], uint, d->nb[l]);
    }
    
    // init belief and input offsets (pointers to big belief and input chunks we
    // allocated above)
    bOffset = 0;
    iOffset = 0;

    // keep track of the max num of beliefs and states.  we need this information
    // to correctly call kernels later
    maxNb = 0;
    maxNs = 0;

    // allocate the input layer offsets.  each node gets an offset from
    // the frame it is presented with.  now computing it indirectly with
    // an array, but there's gotta be a closed-form way of getting the
    // input offset..
    //
    // **note** this is hard-coding a 4-to-1 reduction assuming visual
    // input (2d). we may want 2-to-1 reduction for audio input for
    // future research

    uint n, m;

    uint **inputOffsets;
    uint nInputNodes = pow(4,nl-1);

    MALLOC(inputOffsets, uint *, d->layerSize[0]);
    for( i=0; i < nInputNodes; i++ )
    {
        MALLOC(inputOffsets[i], uint, nInputNodes);
    }

    // get integer sq root of layersize[0]
    uint layerSizeSqRoot = (uint) sqrt( d->layerSize[0] );

    // get integer sq root of ni for lowest layer.  asssumes input is a square.
    uint inputSizeSqRoot = (uint) sqrt( ni );

    // get column size of input image (assuming it is square)
    uint nCols = (uint) sqrt( d->layerSize[0] * ni );

    // calculate offsets.
    uint a, b, innerIdx, bias;
    
    // iterate through rows... (nodes)
    for( i=0, m=0; m < layerSizeSqRoot; m+=2 )
    {
        // iterate through columns... (nodes)
        for( n=0; n < layerSizeSqRoot; n+=2, i+=4 )
        {
            // iterate through rows... (inputs)
            for( innerIdx = 0, a=0; a < inputSizeSqRoot; a++ )
            {
                // iterate through columns... (inputs)
                for( b=0; b < inputSizeSqRoot; b++, innerIdx++ )
                {
                    bias = m*nCols*inputSizeSqRoot + n*inputSizeSqRoot;
                    inputOffsets[i+0][innerIdx] = bias + a*nCols+b;
                    if(nInputNodes > 1){ //case for 1 node 1 layer network
                        inputOffsets[i+1][innerIdx] = bias + a*nCols+b+inputSizeSqRoot;
                        inputOffsets[i+2][innerIdx] = bias + (a+inputSizeSqRoot)*nCols+b;
                        inputOffsets[i+3][innerIdx] = bias + (a+inputSizeSqRoot)*nCols+b+inputSizeSqRoot;
                    }
                }
            }
        }
    }

    uint np = nl > 1 ? nb[1] : 0; //allow 1 layer 1 node networks for testing
    float * sharedCentroids;

    // calculate the state dimensionality (number of inputs + number of beliefs)
    uint ns = ni + nb[0] + np + nc;
    if(isUniform){
        MALLOC(sharedCentroids, float, nb[0]*ns);
    }else{
        sharedCentroids = NULL;
    }
    // initialize zero-layer nodes
    for( n=0, i=0, bOffset = 0; i < d->layerSize[0]; i++, n++ )
    {
        InitNode( 
                    n,
                    d,
                    ni,
                    nb[0],
                    np,
                    nc,
                    ns,
                    starvCoeff,
                    beta,
                    gamma,
                    lambda,
                    temp[0],
                    &d->nodes[n],
                    inputOffsets[n],
                    NULL,
                    &d->belief[bOffset],
                    sharedCentroids
                    );

        // increment belief offset
        bOffset += nb[0];
    }

    // update max belief 
    if( nb[0] > maxNb )
    {
        maxNb = nb[0];
    }

    if( ni + nb[0] + np > maxNs )
    {
        maxNs = ni + nb[0] + np;
    }

    // init the train mask (determines which layers should be training)
    MALLOC(d->layerMask, uint, d->nLayers);
    for( i=0; i < d->nLayers; i++ )
    {
        d->layerMask[i] = 0;
    }

    // initialize the rest of the network
    for( l=1; l < nl; l++ )
    {
        // update max belief
        if( nb[l] > maxNb )
        {
            maxNb = nb[l];
        }

        uint np = (l == nl - 1) ? 0 : nb[l + 1];

        if( 4*nb[l-1] + nb[l] + np > maxNs )
        {
            maxNs = 4*nb[l-1] + nb[l] + np;
        }

        float * sharedCentroids;

        // calculate the state dimensionality (number of inputs + number of beliefs)
        uint ns = nb[l-1]*4 + nb[l] + np + nc;
        if(isUniform){
            MALLOC(sharedCentroids, float, ns*nb[l]);
        }else{
            sharedCentroids = NULL;
        }

        for( i=0; i < d->layerSize[l]; i++, n++ )
        {
            InitNode
                    (   
                        n, 
                        d,
                        nb[l-1]*4,
                        nb[l],
                        np,
                        nc,
                        ns,
                        starvCoeff, 
                        beta, 
                        gamma,
                        lambda,
                        temp[l],
                        &d->nodes[n], 
                        NULL,
                        &d->inputPipeline[iOffset],
                        &d->belief[bOffset],
                        sharedCentroids
                    );
            MALLOC( d->nodes[n].children, Node *, 4 );

            // increment previous belief offset (input to next node)
            iOffset += 4*nb[l-1];

            // increment belief offset (so beliefs are mapped contiguously in memory)
            bOffset += nb[l];
        }
    }
    
    LinkParentBeliefToChildren( d );
    d->maxNb = maxNb;
    d->maxNs = maxNs;


    //setup book keeping to be able to look up nodes by row col and layer

    int layerSize;
    MALLOC( d->nodeRef, int *, d->nLayers );

    // set up layer/row/col references for nodes
    for( i=0, l=0; l < d->nLayers - 1; l++ )
    {
        layerSize = d->layerSize[l];
        layerSizeSqRoot = (uint) sqrt( layerSize );

        MALLOC( d->nodeRef[l], int, layerSize );

        for( m=0; m < layerSizeSqRoot; m+=2 )
        {
            for( n=0; n < layerSizeSqRoot; n+=2, i+=4 )
            {
                d->nodeRef[l][  m   * layerSizeSqRoot + n  ] = i;
                d->nodeRef[l][  m   * layerSizeSqRoot + n+1] = i+1;
                d->nodeRef[l][(m+1) * layerSizeSqRoot + n  ] = i+2;
                d->nodeRef[l][(m+1) * layerSizeSqRoot + n+1] = i+3;
            }
        }
    }

    // set up layer/row/col reference for top node
    MALLOC( d->nodeRef[d->nLayers - 1], int, 1 );
    d->nodeRef[l][0] = d->nNodes - 1;

    

    for( i=0; i < nInputNodes; i++ )
    {
        free(inputOffsets[i]);
    }

    free(inputOffsets);

    return d;
}

void LinkParentBeliefToChildren( Destin *d )
{
    Node *node, *parent;
    uint i, n, l;

    uint parentBias = d->layerSize[0];
    for( n=0, l=0; l < d->nLayers - 1; l++ )
    {
        for( i=0; i < d->layerSize[l]; i++, n++)
        {
            // get structs from device
            node = &d->nodes[n];
            parent = &d->nodes[parentBias + i / 4];

            parent->children[i % 4] = node;

            // update values
            node->parent_pBelief = parent->pBelief;
        }

        parentBias += d->layerSize[l+1];
    }
}

void DestroyDestin( Destin * d )
{
    uint i;

    if(d->isUniform){
        for(i = 0 ; i < d->nLayers; i++)
        {
            //since all nodes in a layer share the same centroids
            //then this only need to free mu once per layer
            FREE(GetNodeFromDestin(d, i, 0,0)->mu);
        }
    }
    
    for( i=0; i < d->nNodes; i++ )
    {
        if(d->isUniform)
        {
            //mu already has been freed so set it to NULL 
            d->nodes[i].mu = NULL;
        }
        
        DestroyNode( &d->nodes[i] );
    }
    FREE(d->temp);
    FREE(d->nb);
    FREE(d->nodes);
    FREE(d->layerMask);
    FREE(d->inputPipeline);
    FREE(d->belief);
    FREE(d->layerSize);
    
    for( i=0; i < d->nLayers; i++ )
    {
        FREE( d->nodeRef[i] );
        FREE( d->sharedCentroidsWinCounts[i] );
    }
    
    FREE( d->sharedCentroidsWinCounts);
    
    FREE(d);
}

// set all nodes to have a uniform belief
void ClearBeliefs( Destin *d )
{
    uint i, n;

    for( n=0; n < d->nNodes; n++ )
    {
        for( i=0; i < d->nodes[n].nb; i++)
        {
            d->nodes[n].pBelief[i] = 1 / (float) d->nodes[n].nb;
        }
    }

    memcpy( d->inputPipeline, d->belief, sizeof(float)*d->nInputPipeline );
}

void SaveDestin( Destin *d, char *filename )
{
    uint i, j;
    Node *nTmp;

    FILE *dFile;

    dFile = fopen(filename, "w");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return;
    }
    //TODO: save the isUniform field
    //TODO: how about the noderef? check that all fields are being saved

    // write destin hierarchy information to disk
    fwrite(&d->nMovements, sizeof(uint), 1, dFile);
    fwrite(&d->nc, sizeof(uint), 1, dFile);
    fwrite(&d->nodes[0].ni, sizeof(uint), 1, dFile);
    fwrite(&d->nLayers, sizeof(uint), 1, dFile);
    fwrite(d->nb, sizeof(uint), d->nLayers, dFile);

    // write destin params to disk
    fwrite(d->temp, sizeof(float), d->nLayers, dFile);
    fwrite(&d->nodes[0].beta, sizeof(float), 1, dFile);
    fwrite(&d->nodes[0].lambda, sizeof(float), 1, dFile);
    fwrite(&d->nodes[0].gamma, sizeof(float), 1, dFile);
    fwrite(&d->nodes[0].starvCoeff, sizeof(float), 1, dFile);

    // write node statistics to disk
    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];

        // write statistics
        fwrite(nTmp->mu, sizeof(float), nTmp->nb*nTmp->ns, dFile);
        fwrite(nTmp->sigma, sizeof(float), nTmp->nb*nTmp->ns, dFile);
        fwrite(nTmp->starv, sizeof(float), nTmp->nb, dFile);
        fwrite(nTmp->nCounts, sizeof(long), nTmp->nb, dFile);
    }

    fclose(dFile);
}

Destin * LoadDestin( Destin *d, char *filename )
{
    FILE *dFile;
    uint i;
    Node *nTmp;

    if( d != NULL )
    {
        fprintf(stderr, "Pointer 0x%p is already initialized, clearing!!\n", d);
        DestroyDestin( d );
    }

    uint nMovements, nc, ni, nl;
    uint *nb;

    float beta, lambda, gamma, starvCoeff;
    float *temp;

    dFile = fopen(filename, "r");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return NULL;
    }

    // read destin hierarchy information from disk
    fread(&nMovements, sizeof(uint), 1, dFile);
    fread(&nc, sizeof(uint), 1, dFile);
    fread(&ni, sizeof(uint), 1, dFile);
    fread(&nl, sizeof(uint), 1, dFile);

    MALLOC(nb, uint, nl);
    MALLOC(temp, float, nl);

    fread(nb, sizeof(uint), nl, dFile);

    // read destin params from disk
    fread(temp, sizeof(float), nl, dFile);
    fread(&beta, sizeof(float), 1, dFile);
    fread(&lambda, sizeof(float), 1, dFile);
    fread(&gamma, sizeof(float), 1, dFile);
    fread(&starvCoeff, sizeof(float), 1, dFile);
    
    bool isUniform = false; //TODO: needs to be included in config file
    d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff, nMovements, false);

    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];

        // load statistics
        fread(nTmp->mu, sizeof(float), nTmp->nb*nTmp->ns, dFile);
        fread(nTmp->sigma, sizeof(float), nTmp->nb*nTmp->ns, dFile);
        fread(nTmp->starv, sizeof(float), nTmp->nb, dFile);
        fread(nTmp->nCounts, sizeof(long), nTmp->nb, dFile);
    }

    return d;
}

// Initialize a node
void InitNode
    (
    uint         nodeIdx,
    Destin *     d,
    uint         ni,
    uint         nb,
    uint         np,
    uint         nc,
    uint         ns,
    float       starvCoeff,
    float       beta,
    float       gamma,
    float       lambda,
    float       temp,
    Node        *node,
    uint        *inputOffsets,
    float       *input_host,
    float       *belief_host,
    float       *sharedCentroids
    )
{

    // Initialize node parameters
    node->d             = d;
    node->nb            = nb;
    node->ni            = ni;
    node->np            = np;
    node->ns            = ns;
    node->nc            = nc;
    node->starvCoeff    = starvCoeff;
    node->beta          = beta;
    node->lambda        = lambda;
    node->gamma         = gamma;
    node->temp          = temp;
    node->winner        = 0;

    if(sharedCentroids == NULL){
        //not uniform so each node gets own centroids
        MALLOC( node->mu, float, nb*ns );
    }else{
        node->mu = sharedCentroids;
    }

    MALLOC( node->sigma, float, nb*ns );
    MALLOC( node->starv, float, nb );
    MALLOC( node->beliefEuc, float, nb );
    MALLOC( node->beliefMal, float, nb );
    MALLOC( node->observation, float, ns );
    MALLOC( node->genObservation, float, ns );
    MALLOC( node->nCounts, long, nb );
    MALLOC( node->delta, float, ns);

    node->children = NULL;

    // point to the block-allocated space
    node->input = input_host;
    node->pBelief = belief_host;

    // copy the input offset for the inputs (should be NULL for non-input nodes)
    if( inputOffsets != NULL )
    {
        MALLOC(node->inputOffsets, uint, ni);
        memcpy(node->inputOffsets, inputOffsets, sizeof(uint) * ni);
    }
    else
    {
        node->inputOffsets = NULL;
    }

    uint i,j;
    for( i=0; i < nb; i++ )
    {
        // init belief (node output)
        node->pBelief[i] = 1 / (float)nb;
        node->beliefEuc[i] = 1 / (float)nb;
        node->beliefMal[i] = 1 / (float)nb;
        node->nCounts[i] = 0;

        // init starv trace to one
        node->starv[i] = 1.0f;

        // init mu and sigma
        for(j=0; j < ns; j++)
        {
            node->mu[i*ns+j] = (float) rand() / (float) RAND_MAX;
            node->sigma[i*ns+j] = INIT_SIGMA;
        }
    }

    for( i=0; i < ns; i++ )
    {
        node->observation[i] = (float) rand() / (float) RAND_MAX;
    }

}

// deallocate the node.
void DestroyNode( Node *n)
{
    //use free here instead of FREE so it doesn't fail on NULL 
    //in case it is part of a uniform destin network which
    //would already have mu freed
    free( n->mu ); 
    
    FREE( n->sigma );
    FREE( n->starv );
    FREE( n->beliefEuc );
    FREE( n->beliefMal );
    FREE( n->observation );
    FREE( n->genObservation );
    FREE( n->nCounts );
    FREE( n->delta );

    if( n->children != NULL )
    {
        FREE( n->children );
    }

    n->children = NULL;

    // if it is a zero-layer node, free the input offset array on the host
    if( n->inputOffsets != NULL)
    {
        FREE(n->inputOffsets);
    }
}

float normrnd(float mean, float stddev) {
    static float n2 = 0.0;
    static int n2_cached = 0;
    if (!n2_cached) {
        float x, y, r;
        do {
            x = 2.0*rand()/RAND_MAX - 1;
            y = 2.0*rand()/RAND_MAX - 1;

            r = x*x + y*y;
        } while (r == 0.0 || r > 1.0);
        {
            float d = sqrt(-2.0*log(r)/r);
            float n1 = x*d;
            n2 = y*d;
            float result = n1*stddev + mean;
            n2_cached = 1;
            return result;
        }
    } else {
        n2_cached = 0;
        return n2*stddev + mean;
    }
}

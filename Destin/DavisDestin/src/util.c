#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "macros.h"
#include "node.h"
#include "destin.h"



void SetLearningStrat(Destin * d, CentroidLearnStrat strategy){
    d->centLearnStrat = strategy;
    switch(strategy){
        case CLS_FIXED:
            d->centLearnStratFunc = &CLS_Fixed;
            break;
        case CLS_DECAY:
            d->centLearnStratFunc = &CLS_Decay;
            break;
        default:
            fprintf(stderr, "Warning: Invalid centroid update strategy enum value %i. Setting null.\n", strategy);
            d->centLearnStratFunc = NULL;
            break;
    }
    return;
}

// create a destin instantiation from a file
//TODO: update to work with uniform destin
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

    // is uniform, i.e. shared centroids
    // 0 = uniform off, 1 = uniform on
    uint iu;
    fscanf(configFile, "%u", &iu);
    bool isUniform = iu == 0 ? false : true;

    // applies boltzman distibution
    // 0 = off, 1 = on
    uint db;
    fscanf(configFile, "%u", &db);
    bool doesBoltzman = db == 0 ? false : true;
    printf("beta: %0.2f. lambda: %0.2f. gamma: %0.2f. starvCoeff: %0.2f\n",beta, lambda, gamma, starvCoeff);
    printf("isUniform: %s. boltzman: %s.", isUniform ? "YES" : "NO", doesBoltzman ? "YES" : "NO");

    newDestin = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff, nMovements, isUniform, doesBoltzman);

    fclose(configFile);

    free(nb);
    free(temp);

    return newDestin;
}

// allocate the input layer offsets.  each node gets an offset from
// the frame it is presented with.  now computing it indirectly with
// an array, but there's gotta be a closed-form way of getting the
// input offset..
//
// **note** this is hard-coding a 4-to-1 reduction assuming visual
// input (2d). we may want 2-to-1 reduction for audio input for
// future research
void CalcNodeInputOffsets(

    Destin * d,
    uint layer,
    uint layer_node_id,
    uint child_layer_offset,
    uint child_input_region_width, //width of the input region in pixels (when layer = 0) or nodes ( for upper layers)
    uint * inputOffsets_out){



    uint    pr, //parent row
            pc, //parent col
            cr, //child row in the entire child layer ( or pixel row if input image)
            cc, //child col in the entire child layer ( or pixel col if input image)
            clnw,//child layer node width
            cirw = child_input_region_width,
            clnb,//child layer nb ( size of output belief vector for each child node)
            plw = d->layerWidth[layer], //parent layer width
            i, j, k;//parent layer width

    if(layer == 0){
        // If its the input layer then pretend each pixel of the 4x4 input image( assuming ni = 16)
        // is a node (in layer below layer 0) with nb = 1, then the nodes in
        // layer 0 take in 4x4 prentend child nodes instead of just 2x2 nodes like in the upper layers
        clnb = 1;
        clnw = plw * cirw ; // Each layer 0 node takes in a square region of width 4 pretend nodes (i.e pixels)
                            // so this is equal to the image width.
    }else{
        clnb = d->nb[layer - 1],//child layer nb
        clnw = plw * cirw; //the width of the entire child layer in nodes
    }

    pr = layer_node_id / plw; // Convert nodeid into row, col coordinates
    pc = layer_node_id % plw;

    cr = pr * cirw;
    cc = pc * cirw;
    uint cos[cirw * cirw]; // Child output start ( start of child node belief output vector)
    uint child_region_row, // The row in the 4x4 or 2x2 input region
         child_region_col; // The col in the 4x4 or 2x2 input region

    i = 0;
    for(child_region_row = 0 ;child_region_row < cirw; child_region_row++){
        for(child_region_col = 0 ; child_region_col < cirw; child_region_col++){
            cos[i] = child_layer_offset + clnb * ( (cr + child_region_row) * clnw + (cc + child_region_col) );
            i++;
        }
    }


    k=0;
    for(i = 0 ; i < cirw * cirw ; i++){
        for(j = 0 ; j < clnb ; j++){
            inputOffsets_out[k++] = cos[i] + j;
        }
    }

}

Destin * InitDestin( uint ni, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma, float *temp, float starvCoeff, uint nMovements, bool isUniform, bool doesBoltzman )
{
    uint nNodes, nInputPipeline;
    uint i, l, nBeliefs, maxNb, maxNs;
    size_t bOffset ;


    Destin *d;

    // initialize a new Destin object
    MALLOC(d, Destin, 1);

    d->nNodes = 0;
    d->nLayers = nl;

    d->nMovements = nMovements;
    d->isUniform = isUniform;
    d->doesBoltzman = doesBoltzman;
    d->muSumSqDiff = 0;

    SetLearningStrat(d, CLS_DECAY);

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
    MALLOC(d->layerWidth, uint, nl);

    nNodes = 0;
    nBeliefs = 0;
    for( i=0, l=nl ; l != 0; l--, i++ )
    {
        d->layerSize[i] = 1 << 2*(l-1);
        d->layerWidth[i] = (uint)sqrt(d->layerSize[i]);

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

    if(isUniform){
        // allocate for each layer an array of size number = n centroids for that layer
        // that counts how many nodes in a layer pick the given centroid as winner.
        MALLOC(d->uf_winCounts, uint *, d->nLayers);
        MALLOC(d->uf_persistWinCounts, long *, d->nLayers);

        // layer shared centroid starvation vectors
        MALLOC(d->uf_starv, float *, d->nLayers);

        for(l = 0 ; l < d->nLayers ; l++){
            MALLOC( d->uf_winCounts[l], uint, d->nb[l]);
            MALLOC( d->uf_persistWinCounts[l], long, d->nb[l] );
            MALLOC( d->uf_starv[l], float, d->nb[l]);

            for(i = 0 ; i < d->nb[l]; i++){
                d->uf_persistWinCounts[l][i] = 0;
                d->uf_starv[l][i] = 1;
            }
        }
    }

    // init belief and input offsets (pointers to big belief and input chunks we
    // allocated above)
    bOffset = 0;

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
        MALLOC(inputOffsets[i], uint, ni);
        CalcNodeInputOffsets(d, 0, i, 0, (uint)sqrt(ni), inputOffsets[i]);
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

    if(isUniform){
        //used to calculate the shared centroid delta averages
        MALLOC(d->uf_avgDelta, float *, d->nLayers);
        MALLOC(d->uf_avgDelta[0], float, ns*nb[0]);//the rest are allocated later

        MALLOC(d->uf_sigma, float *, d->nLayers);
        MALLOC(d->uf_sigma[0], float, ns*nb[0]);//the rest are allocated later
    }

    // initialize zero-layer nodes
    for( n=0, i=0, bOffset = 0; i < d->layerSize[0]; i++, n++ )
    {
        InitNode( 
                    n,
                    d,
                    0,
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

    uint child_layer_offset = 0;
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
            MALLOC(d->uf_avgDelta[l], float, ns*nb[l]);
            MALLOC(sharedCentroids, float, ns*nb[l]);
            MALLOC(d->uf_sigma[l], float, ns*nb[l]);
        }else{
            sharedCentroids = NULL;
        }


        uint * inputOffsets_temp;
        MALLOC(inputOffsets_temp, uint, (nb[l - 1] * 4));

        //iterate over nodes in this layer
        for( i=0; i < d->layerSize[l]; i++, n++ )
        {
            CalcNodeInputOffsets(d, l, i, child_layer_offset, 2, inputOffsets_temp);
            InitNode(
                        n, 
                        d,
                        l,
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
                        inputOffsets_temp,
                        d->inputPipeline,
                        &d->belief[bOffset],
                        sharedCentroids
                    );
            MALLOC( d->nodes[n].children, Node *, 4 );

            // increment previous belief offset (input to next node)
            //iOffset += 4*nb[l-1];

            // increment belief offset (so beliefs are mapped contiguously in memory)
            bOffset += nb[l];
        }//next node

        child_layer_offset += 4*nb[l-1] * d->layerSize[l];
        FREE(inputOffsets_temp);

    }//next layer
    
    LinkParentBeliefToChildren( d );
    d->maxNb = maxNb;
    d->maxNs = maxNs;



    for( i=0; i < nInputNodes; i++ )
    {
        free(inputOffsets[i]);
    }

    free(inputOffsets);
    return d;
}

void LinkParentBeliefToChildren( Destin *d )
{
    uint l, cr, cc, pr, pc, child_layer_width;
    Node * child_node;
    Node * parent_node;

    for(l = 0 ; l < d->nLayers - 1 ; l++){
        child_layer_width = d->layerWidth[l];
        for(cr = 0 ; cr < child_layer_width; cr++){
            pr = cr / 2;
            for(cc = 0; cc < child_layer_width ; cc++){
                pc = cc / 2;
                child_node = GetNodeFromDestin(d, l, cr, cc);
                parent_node = GetNodeFromDestin(d, l+1, pr, pc);
                child_node->parent_pBelief = parent_node->pBelief;
                parent_node->children[(cr % 2) * 2 + cc % 2] = child_node;
            }
        }
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

            FREE(d->uf_avgDelta[i]);
            FREE(d->uf_winCounts[i]); //TODO: should be condionally alloced and delloc based of if using uniform destin
            FREE(d->uf_persistWinCounts[i]);
            FREE(d->uf_sigma[i]);
            FREE(d->uf_starv[i]);
        }
        FREE(d->uf_avgDelta);
        FREE(d->uf_winCounts);
        FREE(d->uf_persistWinCounts);
        FREE(d->uf_sigma);
        FREE(d->uf_starv);
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
    FREE(d->layerWidth);
    
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
    uint i, l;
    Node *nTmp;

    FILE *dFile;

    dFile = fopen(filename, "w");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return;
    }

    // write destin hierarchy information to disk
    fwrite(&d->nMovements,  sizeof(uint), 1,            dFile);
    fwrite(&d->nc,          sizeof(uint), 1,            dFile);
    fwrite(&d->nodes[0].ni, sizeof(uint), 1,            dFile);
    fwrite(&d->nLayers,     sizeof(uint), 1,            dFile);
    fwrite(&d->isUniform,   sizeof(bool), 1,            dFile);
    fwrite(&d->doesBoltzman,sizeof(bool), 1,            dFile);
    fwrite(d->nb,           sizeof(uint), d->nLayers,   dFile);

    // write destin params to disk
    fwrite(d->temp,                 sizeof(float),          d->nLayers,  dFile);
    fwrite(&d->nodes[0].beta,       sizeof(float),          1,           dFile); //TODO consider moving these constants to the destin struc
    fwrite(&d->nodes[0].lambda,     sizeof(float),          1,           dFile);
    fwrite(&d->nodes[0].gamma,      sizeof(float),          1,           dFile);
    fwrite(&d->nodes[0].starvCoeff, sizeof(float),          1,           dFile);
    fwrite(&d->centLearnStrat,       sizeof(CentroidLearnStrat),    1,           dFile);

    //write belief states
    fwrite(d->inputPipeline, sizeof(float), d->nInputPipeline, dFile);
    fwrite(d->belief,        sizeof(float), d->nBeliefs,       dFile);

    // write node statistics to disk

    if(d->isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0); //get the 0th node of the layer
            fwrite(nTmp->mu,                    sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fwrite(d->uf_avgDelta[l],           sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fwrite(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],               dFile);
            fwrite(d->uf_sigma[l],              sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fwrite(d->uf_starv[l],              sizeof(float),  d->nb[l],               dFile);
            fwrite(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],               dFile);
        }

    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];

            // write statistics
            fwrite(nTmp->mu,        sizeof(float),  nTmp->nb*nTmp->ns,  dFile);
            fwrite(nTmp->sigma,     sizeof(float),  nTmp->nb*nTmp->ns,  dFile);
            fwrite(nTmp->starv,     sizeof(float),  nTmp->nb,           dFile);
            fwrite(nTmp->nCounts,   sizeof(long),   nTmp->nb,           dFile);
        }
    }


    fclose(dFile);
}

Destin * LoadDestin( Destin *d, char *filename )
{
    FILE *dFile;
    uint i, l;
    Node *nTmp;

    if( d != NULL )
    {
        fprintf(stderr, "Pointer 0x%p is already initialized, clearing!!\n", d);
        DestroyDestin( d );
    }

    uint nMovements, nc, ni, nl;
    bool isUniform, doesBoltzman;
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
    fread(&nMovements,  sizeof(uint), 1, dFile);
    fread(&nc,          sizeof(uint), 1, dFile);
    fread(&ni,          sizeof(uint), 1, dFile);
    fread(&nl,          sizeof(uint), 1, dFile);
    fread(&isUniform,   sizeof(bool), 1, dFile);
    fread(&doesBoltzman,sizeof(bool), 1, dFile);

    MALLOC(nb, uint, nl);
    MALLOC(temp, float, nl);

    fread(nb, sizeof(uint), nl, dFile);

    // read destin params from disk
    fread(temp, sizeof(float), nl, dFile);
    fread(&beta, sizeof(float), 1, dFile);
    fread(&lambda, sizeof(float), 1, dFile);
    fread(&gamma, sizeof(float), 1, dFile);
    fread(&starvCoeff, sizeof(float), 1, dFile);

    d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff, nMovements, isUniform, doesBoltzman);

    fread(&d->centLearnStrat,sizeof(CentroidLearnStrat),   1,                 dFile);
    SetLearningStrat(d, d->centLearnStrat);

    fread(d->inputPipeline, sizeof(float),         d->nInputPipeline, dFile);
    fread(d->belief,        sizeof(float),         d->nBeliefs,       dFile);

    if(isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0);
            fread(nTmp->mu,                    sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fread(d->uf_avgDelta[l],           sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fread(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],               dFile);
            fread(d->uf_sigma[l],              sizeof(float),  d->nb[l] * nTmp->ns,    dFile);
            fread(d->uf_starv[l],              sizeof(float),  d->nb[l],               dFile);
            fread(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],               dFile);
        }

    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];

            // load statistics
            fread(nTmp->mu, sizeof(float), nTmp->nb*nTmp->ns, dFile);
            fread(nTmp->sigma, sizeof(float), nTmp->nb*nTmp->ns, dFile);
            fread(nTmp->starv, sizeof(float), nTmp->nb, dFile);
            fread(nTmp->nCounts, sizeof(long), nTmp->nb, dFile);
        }
    }

    return d;
}

// Initialize a node
void InitNode
    (
    uint         nodeIdx,
    Destin *     d,
    uint         layer,
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
    node->nIdx          = nodeIdx;
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
    node->layer         = layer;

    if(sharedCentroids == NULL){
        //not uniform so each node gets own centroids
        MALLOC( node->mu, float, nb*ns );
    }else{
        node->mu = sharedCentroids;
    }

    MALLOC( node->beliefEuc, float, nb );
    MALLOC( node->beliefMal, float, nb );
    MALLOC( node->observation, float, ns );
    MALLOC( node->genObservation, float, ns );

    if(d->isUniform){
        //uniform destin uses shared counts
        node->starv = d->uf_starv[layer];
        node->nCounts = NULL;
        node->sigma = NULL;
    }else{
        MALLOC( node->starv, float, nb );
        MALLOC( node->nCounts, long, nb );
        MALLOC( node->sigma, float, nb*ns );
    }

    MALLOC( node->delta, float, ns);

    node->children = NULL;

    // point to the block-allocated space
    node->input = input_host;
    uint i,j;
    if(input_host != NULL){
        for(i = 0 ; i < ni ; i++){
            node->input[i] = 0.5; //prevent nans caused by uninitialized memory
        }
    }

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

    for( i=0; i < nb; i++ )
    {
        // init belief (node output)
        node->pBelief[i] = 1 / (float)nb;
        node->beliefEuc[i] = 1 / (float)nb;
        node->beliefMal[i] = 1 / (float)nb;

        if(!d->isUniform){
            node->nCounts[i] = 0;
            // init starv trace to one
            node->starv[i] = 1.0f;
        }

        // init mu and sigma
        for(j=0; j < ns; j++)
        {
            node->mu[i*ns+j] = (float) rand() / (float) RAND_MAX;
            if(d->isUniform){
                //TODO: all the nodes in the layer are initing the same shared sigma vectors redundantly, may
                //want to initialize this outside of the node
                node->d->uf_sigma[layer][i * ns + j] = INIT_SIGMA;
            }else{
                node->sigma[i*ns+j] = INIT_SIGMA;
            }
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
    
    if(!n->d->isUniform){
        FREE( n->mu );
        FREE( n->sigma );
        FREE( n->nCounts );
        FREE( n->starv );
    }

    FREE( n->beliefEuc );
    FREE( n->beliefMal );
    FREE( n->observation );
    FREE( n->genObservation );

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

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

#include "opencv/cv.h"
#include "opencv/highgui.h"

#include "macros.h"
#include "node.h"
#include "destin.h"
#include "array.h"
#include "centroid.h"

// Defines the version of destin structs that it can load and save.
// This is to try to prevent loading older incompatible destin structs from file
// and causing an unknown crash.
// This value should be incremented by the developer when the order of the destin struc
// fields are moved around or if fields are added, removed, ect.
#define SERIALIZE_VERSION 6

void initializeDestinParameters(uint *nb, bool isUniform, uint *nci, int extRatio, uint nl, uint nMovements,
                                Destin* d, uint nc, float *temp, float freqCoeff, float freqTreshold);

void SetLearningStrat(Destin * d, CentroidLearnStrat strategy){
    d->centLearnStrat = strategy;
    switch(strategy){
        case CLS_FIXED:
            d->centLearnStratFunc = &CLS_Fixed;
            break;
        case CLS_DECAY:
            d->centLearnStratFunc = &CLS_Decay;
            break;
        case CLS_DECAY_c1:
            d->centLearnStratFunc = &CLS_Decay_c1;
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
    int fscanfResult;

    FILE *configFile;
    // open file
    configFile = fopen(filename, "r");
    if( !configFile ) {
        fprintf(stderr, "Cannot open config file %s\n", filename);
        exit(1);
    }
    
    uint nl, nc, i, nMovements;
    uint *nci, *nb;
    float beta, lambda, gamma, starvCoeff, freqCoeff, freqTreshold;
    float *temp;

    // parse config file

    // get number of movements per digit
    fscanfResult = fscanf(configFile, "%d", &nMovements);
    printf("nMovements: %d\n", nMovements);

    // get number of distinct classes
    fscanfResult = fscanf(configFile, "%d", &nc);
    printf("# classes: %d\n", nc);

    // get number of layers
    fscanfResult = fscanf(configFile, "%d", &nl);
    nb = (uint *) malloc(sizeof(uint) * nl);
    nci = (uint *) malloc(sizeof(uint) * nl);
    temp = (float *) malloc(sizeof(float) * nl);

    printf("# layers: %d\n", nl);

    // get layer beliefs and temps
    for(i=0; i < nl; i++) {
        fscanfResult = fscanf(configFile, "%d %d %f", &nci[i], &nb[i], &temp[i]);
        printf("\t%d %d %f\n", nci[i], nb[i], temp[i]);
    }

    // get coeffs
    fscanfResult = fscanf(configFile, "%f", &beta);
    fscanfResult = fscanf(configFile, "%f", &lambda);
    fscanfResult = fscanf(configFile, "%f", &gamma);
    fscanfResult = fscanf(configFile, "%f", &starvCoeff);
    fscanfResult = fscanf(configFile, "%f", &freqCoeff);
    fscanfResult = fscanf(configFile, "%f", &freqTreshold);

    // is uniform, i.e. shared centroids
    // 0 = uniform off, 1 = uniform on
    uint iu;
    fscanfResult = fscanf(configFile, "%u", &iu);
    bool isUniform = iu == 0 ? false : true;

    //TODO: set learning strat and belief transform in config
    //TODO: fix test config file
    // applies boltzman distibution
    // 0 = off, 1 = on
    char bts[80]; //belief transform string

    fscanfResult = fscanf(configFile, "%80s", bts);

    BeliefTransformEnum bte = BeliefTransform_S_to_E(bts);

    printf("beta: %0.2f. lambda: %0.2f. gamma: %0.2f. starvCoeff: %0.2f\n",beta, lambda, gamma, starvCoeff);
    printf("freqCoeff: %0.3f. freqTreshold: %0.3f. ", freqCoeff, freqTreshold);
    printf("isUniform: %s. belief transform: %s.\n", isUniform ? "YES" : "NO", bts);

    newDestin = InitDestin(nci, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff,
                           freqCoeff, freqTreshold, nMovements, isUniform, 1);
    SetBeliefTransform(newDestin, bte);

    fclose(configFile);

    free(nb);
    free(temp);

    return newDestin;
}

/** Calculates input offsets for square shape nodes from layer 0
 * @param layerWidth   - width of layer 0
 * @param nIdx - node index in the layer 0
 * @param ni   - dimensionality of input, should be square number
 * @param inputOffsets - pointer to preallocated uint array which will hold calculated offsets.
 */
void CalcSquareNodeInputOffsets(uint layerWidth, uint nIdx, uint ni, uint * inputOffsets)
{
    uint zr, zc;                 // zero layer row, col coordinates
    uint ir, ic;                 // input layer row, col coordinates
    uint inputWidth = (uint) sqrt(ni);
    uint inputLayerWidth = layerWidth * inputWidth;   // input layer width

    zr = nIdx / layerWidth;      // Convert node index into row, col coordinates
    zc = nIdx % layerWidth;

    ir = zr * inputWidth;        // Convert coordinates into input layer row, col coordinates
    ic = zc * inputWidth;

    uint row, col;
    uint i = 0;
    for (row = 0; row < inputWidth; row++)
    {
        for (col = 0; col < inputWidth; col++)
        {
            inputOffsets[i] = (ir + row) * inputLayerWidth + (ic + col);
            i++;
        }
    }
};

Destin * InitDestin( uint *nci, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma, float *temp,
                     float starvCoeff, float freqCoeff, float freqTreshold, uint nMovements, bool isUniform, int extRatio)
{
    uint i, j, l, maxNb, maxNs;
    Destin *d;

    // initialize a new Destin object
    MALLOC(d, Destin, 1);

    initializeDestinParameters(nb, isUniform, nci, extRatio, nl, nMovements, d, nc, temp, freqCoeff, freqTreshold);

    // keep track of the max num of beliefs and states.  we need this information
    // to correctly call kernels later
    maxNb = 0;
    maxNs = 0;

    uint n = 0;

    // initialize the rest of the network
    for( l=0; l < nl; l++ )
    {
        // update max belief
        if( nb[l] > maxNb )
        {
            maxNb = nb[l];
        }

        uint np = ((l + 1 == nl) ? 0 : nb[l+1]);
        uint ni = (l == 0 ? nci[0] : nci[l] * nb[l-1]);
        uint ns = nb[l] + np + nc + ((l == 0) ? ni*extRatio : ni);

        if (ns > maxNs)
        {
            maxNs = ns;
        }

        if(isUniform){
            InitUniformCentroids(d, l, ni, nb[l], np, ns);
        }

        uint inputOffsets[ni];
        for( i=0; i < d->layerSize[l]; i++, n++ )
        {
            if (l == 0)
            {
                CalcSquareNodeInputOffsets(d->layerWidth[0], i, ni, inputOffsets);
            }

            InitNode(
                        n,
                        d,
                        l,
                        ni,
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
                        (l > 0 ? NULL : inputOffsets),
                        (l > 0 ? nci[l] : 0)
                    );
        }//next node
    }//next layer

    LinkParentsToChildren( d );
    d->maxNb = maxNb;
    d->maxNs = maxNs;

    return d;
}

//TODO: make this a private function
void initializeDestinParameters(uint *nb, bool isUniform, uint *nci, int extRatio, uint nl, uint nMovements, Destin* d,
                                uint nc, float *temp, float freqCoeff, float freqTreshold)
{
    uint l;
    int i; // must be signed int
    d->serializeVersion = SERIALIZE_VERSION;
    d->nNodes = 0;
    d->nLayers = nl;

    d->nMovements = nMovements;
    d->isUniform = isUniform;
    d->muSumSqDiff = 0;

    SetLearningStrat(d, CLS_DECAY);
    SetBeliefTransform(d, DST_BT_NONE);

    d->fixedLearnRate = 0.1;
    d->freqCoeff = freqCoeff;
    d->freqTreshold = freqTreshold;

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

    MALLOC(d->nci, uint, nl);
    for ( l = 0; l < nl; l++)
    {
        // make sure layer sizes are square
        d->nci[l] = (uint)sqrt(nci[l])*(uint)sqrt(nci[l]);
    }

    MALLOC(d->layerSize, uint, nl);
    MALLOC(d->layerNodeOffsets, uint, nl);
    MALLOC(d->layerWidth, uint, nl);

    // init the train mask (determines which layers should be training)
    MALLOC(d->layerMask, uint, d->nLayers);
    for( i=0; i < d->nLayers; i++ )
    {
        d->layerMask[i] = 0;
    }

    d->layerSize[d->nLayers-1] = 1;
    d->layerWidth[d->nLayers-1] = 1;
    for ( i=d->nLayers-2; i >= 0; i-- )
    {
        d->layerSize[i] = d->layerSize[i+1]*d->nci[i+1];
        d->layerWidth[i] = d->layerWidth[i+1]*(uint)sqrt(d->nci[i+1]);
    }

    uint nNodes = 0;
    for( i=0; i < d->nLayers; i++ )
    {
        d->layerNodeOffsets[i] = nNodes;
        nNodes += d->layerSize[i];
    }
    d->inputImageSize = d->layerSize[0] * d->nci[0];
    d->extRatio = extRatio;

    d->nNodes = nNodes;

    // allocate node pointers on host
    MALLOC(d->nodes, Node, nNodes);

    if(isUniform){
        // Allocate for each layer an array of size number = n centroids for that layer
        // that counts how many nodes in a layer pick the given centroid as winner.
        MALLOC(d->uf_winCounts, uint *, d->nLayers);
        MALLOC(d->uf_persistWinCounts, long *, d->nLayers);
        MALLOC(d->uf_winFreqs, float *, d->nLayers);

        // Used to calculate the shared centroid delta averages
        MALLOC(d->uf_avgDelta, float **, d->nLayers);
        MALLOC(d->uf_mu, float **, d->nLayers);
        MALLOC(d->uf_sigma, float **, d->nLayers);

        // layer shared centroid starvation vectors
        MALLOC(d->uf_starv, float *, d->nLayers);

        // 2013.6.13
        // CZT
        MALLOC(d->uf_persistWinCounts_detailed, long *, d->nLayers);
        // 2013.7.4
        // CZT: uf_absvar, very similar to uf_sigma;
        MALLOC(d->uf_absvar, float *, d->nLayers);
        // 2013.7.18
        MALLOC(d->uf_avgSquaredDelta, float **, d->nLayers);
        MALLOC(d->uf_avgAbsDelta, float *, d->nLayers);
    }
}

void LinkParentsToChildren( Destin *d )
{
    uint l, cr, cc, pr, pc, child_layer_width;
    uint piw;   // parent input width
    Node * child_node;
    Node * parent_node;

    for(l = 0 ; l + 1 < d->nLayers ; l++){
        child_layer_width = d->layerWidth[l];
        piw = (uint) sqrt(d->nci[l+1]);
        for(cr = 0 ; cr < child_layer_width; cr++){
            pr = cr / piw;
            for(cc = 0; cc < child_layer_width ; cc++){
                pc = cc / piw;
                child_node = GetNodeFromDestin(d, l, cr, cc);
                parent_node = GetNodeFromDestin(d, l+1, pr, pc);
                parent_node->children[(cr % piw) * piw + cc % piw] = child_node;
                child_node->parent = parent_node;
            }
        }
    }
}

void DestroyDestin( Destin * d )
{
    uint i, j;

    if(d->isUniform){
        for(i = 0; i < d->nLayers; i++)
        {
            for (j = 0; j < d->nb[i]; j++)
            {
                FREE(d->uf_mu[i][j]);
                FREE(d->uf_sigma[i][j]);
                FREE(d->uf_avgDelta[i][j]);
                FREE(d->uf_avgSquaredDelta[i][j]);
            }
            FREE(d->uf_mu[i]);
            FREE(d->uf_sigma[i]);
            FREE(d->uf_absvar[i]);
            FREE(d->uf_winCounts[i]);
            FREE(d->uf_winFreqs[i]);
            FREE(d->uf_persistWinCounts[i]);
            FREE(d->uf_persistWinCounts_detailed[i]);
            FREE(d->uf_avgDelta[i]);
            FREE(d->uf_avgAbsDelta[i]);
            FREE(d->uf_avgSquaredDelta[i]);
            FREE(d->uf_starv[i]);
        }
        FREE(d->uf_mu);
        FREE(d->uf_sigma);
        FREE(d->uf_absvar);
        FREE(d->uf_winCounts);
        FREE(d->uf_winFreqs);
        FREE(d->uf_persistWinCounts);
        FREE(d->uf_persistWinCounts_detailed);
        FREE(d->uf_avgDelta);
        FREE(d->uf_avgAbsDelta);
        FREE(d->uf_avgSquaredDelta);
        FREE(d->uf_starv);
    }
    
    for( i=0; i < d->nNodes; i++ )
    {
        DestroyNode( &d->nodes[i] );
    }

    FREE(d->temp);
    FREE(d->nb);
    FREE(d->nci);
    FREE(d->nodes);
    FREE(d->layerMask);
    FREE(d->layerSize);
    FREE(d->layerNodeOffsets);
    FREE(d->layerWidth);
    FREE(d->inputLabel);
    FREE(d);
}

// copy previous beliefs into nodes output
void CopyOutputBeliefs( Destin *d )
{
    uint i, n;

    for( n=0; n < d->nNodes; n++ )
    {
        for( i=0; i < d->nodes[n].nb; i++)
        {
            d->nodes[n].outputBelief[i] = d->nodes[n].belief[i];
        }
    }
}

// set all nodes to have a uniform belief
void ClearBeliefs( Destin *d )
{
    uint i, n;

    for( n=0; n < d->nNodes; n++ )
    {
        for( i=0; i < d->nodes[n].nb; i++)
        {
            d->nodes[n].belief[i] = 1 / (float) d->nodes[n].nb;
            d->nodes[n].outputBelief[i] = d->nodes[n].belief[i];
        }
    }

    CopyOutputBeliefs( d );
}

void SaveDestin( Destin *d, char *filename )
{
    uint i, j, l;
    Node *nTmp;

    FILE *dFile;

    dFile = fopen(filename, "w");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return;
    }

    fwrite(&d->serializeVersion, sizeof(uint), 1,       dFile);

    // write destin hierarchy information to disk
    fwrite(&d->nMovements,  sizeof(uint), 1,            dFile);
    fwrite(&d->nc,          sizeof(uint), 1,            dFile);
    fwrite(&d->nLayers,     sizeof(uint), 1,            dFile);
    fwrite(&d->isUniform,   sizeof(bool), 1,            dFile);
    fwrite(d->nb,           sizeof(uint), d->nLayers,   dFile);
    fwrite(d->nci,          sizeof(uint), d->nLayers,   dFile);

    // write destin params to disk
    fwrite(d->temp,                 sizeof(float),              d->nLayers,  dFile);
    fwrite(&d->nodes[0].beta,       sizeof(float),              1,           dFile); //TODO consider moving these constants to the destin struc
    fwrite(&d->nodes[0].nLambda,    sizeof(float),              1,           dFile);
    fwrite(&d->nodes[0].gamma,      sizeof(float),              1,           dFile);
    fwrite(&d->nodes[0].starvCoeff, sizeof(float),              1,           dFile);
    fwrite(&d->freqCoeff,           sizeof(float),              1,           dFile);
    fwrite(&d->freqTreshold,        sizeof(float),              1,           dFile);
    fwrite(&d->extRatio,            sizeof(int),                1,           dFile);

    fwrite(&d->centLearnStrat,      sizeof(CentroidLearnStrat), 1,           dFile);
    fwrite(&d->beliefTransform,     sizeof(BeliefTransformEnum),1,           dFile);
    fwrite(&d->fixedLearnRate,      sizeof(float),              1,           dFile);

    // write node belief states
    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];
        fwrite(nTmp->belief,           sizeof(float), nTmp->nb, dFile);
        fwrite(nTmp->outputBelief,     sizeof(float), nTmp->nb, dFile);
    }

    // write node statistics to disk
    if(d->isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0); //get the 0th node of the layer
            for (i = 0; i < d->nb[l]; i++){
                fwrite(d->uf_mu[l][i],          sizeof(float),  nTmp->ns,    dFile);
                fwrite(d->uf_sigma[l][i],       sizeof(float),  nTmp->ns,    dFile);
                fwrite(d->uf_avgDelta[l][i],    sizeof(float),  nTmp->ns,    dFile);
            }
            fwrite(d->uf_absvar[l],             sizeof(float),  nTmp->ns,               dFile);
            fwrite(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],               dFile);
            fwrite(d->uf_winFreqs[l],           sizeof(float),  d->nb[l],               dFile);
            fwrite(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],               dFile);
            fwrite(d->uf_persistWinCounts_detailed[l],   sizeof(long),   d->nb[l],      dFile);
            fwrite(d->uf_starv[l],              sizeof(float),  d->nb[l],               dFile);
        }
    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];
            // write statistics
            for (j = 0; j < nTmp->nb; j++){
                fwrite(nTmp->mu[j],          sizeof(float),  nTmp->ns,    dFile);
                fwrite(nTmp->sigma[j],       sizeof(float),  nTmp->ns,    dFile);
            }
            fwrite(nTmp->starv,     sizeof(float),  nTmp->nb,           dFile);
            fwrite(nTmp->nCounts,   sizeof(long),   nTmp->nb,           dFile);
        }
    }

    fclose(dFile);
}

Destin * LoadDestin( Destin *d, const char *filename )
{
    FILE *dFile;
    uint i, j, l;
    Node *nTmp;
    size_t freadResult;

    if( d != NULL )
    {
        fprintf(stderr, "Pointer 0x%p is already initialized, clearing!!\n", d);
        DestroyDestin( d );
    }

    uint serializeVersion;
    uint nMovements, nc, nl;
    bool isUniform;
    int extendRatio; //TODO: make this a uint
    uint *nci, *nb;

    float beta, lambda, gamma, starvCoeff, freqCoeff, freqTreshold;
    float *temp;

    dFile = fopen(filename, "r");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return NULL;
    }

    freadResult = fread(&serializeVersion, sizeof(uint), 1, dFile);
    if(serializeVersion != SERIALIZE_VERSION){
        fprintf(stderr, "Error: can't load %s because its version is %i, and we're expecting %i\n", filename, serializeVersion, SERIALIZE_VERSION);
        return NULL;
    }

    // read destin hierarchy information from disk
    freadResult = fread(&nMovements,  sizeof(uint), 1, dFile);
    freadResult = fread(&nc,          sizeof(uint), 1, dFile);
    freadResult = fread(&nl,          sizeof(uint), 1, dFile);
    freadResult = fread(&isUniform,   sizeof(bool), 1, dFile);

    MALLOC(nb, uint, nl);
    MALLOC(nci, uint, nl);
    MALLOC(temp, float, nl);

    freadResult = fread(nb, sizeof(uint), nl, dFile);
    freadResult = fread(nci, sizeof(uint), nl, dFile);

    // read destin params from disk
    freadResult = fread(temp,         sizeof(float),    nl,  dFile);
    freadResult = fread(&beta,        sizeof(float),    1,   dFile);
    freadResult = fread(&lambda,      sizeof(float),    1,   dFile);
    freadResult = fread(&gamma,       sizeof(float),    1,   dFile);
    freadResult = fread(&starvCoeff,  sizeof(float),    1,   dFile);
    freadResult = fread(&freqCoeff,   sizeof(float),    1,   dFile);
    freadResult = fread(&freqTreshold, sizeof(float),   1,   dFile);
    freadResult = fread(&extendRatio, sizeof(int),      1,   dFile);

    d = InitDestin(nci, nl, nb, nc, beta, lambda, gamma, temp, starvCoeff,
                   freqCoeff, freqTreshold, nMovements, isUniform, extendRatio);

    // temporary arrays were copied in InitDestin
    FREE(nb); nb = NULL;
    FREE(temp); temp = NULL;


    freadResult = fread(&d->centLearnStrat,sizeof(CentroidLearnStrat),   1,         dFile);
    SetLearningStrat(d, d->centLearnStrat);

    freadResult = fread(&d->beliefTransform, sizeof(BeliefTransformEnum),1,         dFile);
    SetBeliefTransform(d, d->beliefTransform);

    freadResult = fread(&d->fixedLearnRate,sizeof(float),                1,         dFile);

    // load node belief states
    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];
        freadResult = fread(nTmp->belief,       sizeof(float), nTmp->nb, dFile);
        freadResult = fread(nTmp->outputBelief, sizeof(float), nTmp->nb, dFile);
    }

    if(isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0);
            for(i = 0 ; i < d->nb[l]; i++){
                freadResult = fread(d->uf_mu[l][i],          sizeof(float),  nTmp->ns,    dFile);
                freadResult = fread(d->uf_sigma[l][i],       sizeof(float),  nTmp->ns,    dFile);
                freadResult = fread(d->uf_avgDelta[l][i],       sizeof(float),  nTmp->ns,    dFile);
            }
            freadResult = fread(d->uf_absvar[l],             sizeof(float),  nTmp->ns,    dFile);
            freadResult = fread(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],    dFile);
            freadResult = fread(d->uf_winFreqs[l],           sizeof(float),  d->nb[l],    dFile);
            freadResult = fread(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],    dFile);
            freadResult = fread(d->uf_persistWinCounts_detailed[l],   sizeof(long),   d->nb[l],    dFile);
            freadResult = fread(d->uf_starv[l],              sizeof(float),  d->nb[l],    dFile);
        }

    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];

            // load statistics
            for(j = 0 ; j < nTmp->nb; j++){
                freadResult = fread(nTmp->mu[j],    sizeof(float),  nTmp->ns,    dFile);
                freadResult = fread(nTmp->sigma[j], sizeof(float),  nTmp->ns,    dFile);
            }
            freadResult = fread(nTmp->starv, sizeof(float), nTmp->nb, dFile);
            freadResult = fread(nTmp->nCounts, sizeof(long), nTmp->nb, dFile);
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
    uint        childNumber
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
    node->nLambda       = lambda;
    node->gamma         = gamma;
    node->temp          = temp;
    node->winner        = 0;
    node->layer         = layer;
    node->childNumber   = childNumber;

    int relativeIndex = nodeIdx - d->layerNodeOffsets[layer];
    node->row           = relativeIndex / d->layerWidth[layer];
    node->col           = relativeIndex - node->row * d->layerWidth[layer];

    uint i,j;

    if(d->isUniform){
        node->mu = d->uf_mu[layer];
        node->starv = d->uf_starv[layer];
        node->nCounts = NULL;
        node->sigma = NULL;
    }else{
        MALLOCV(node->mu, float *, nb);
        MALLOCV(node->sigma, float *, nb);
        MALLOCV(node->starv, float, nb);
        MALLOCV(node->nCounts, long, nb);
        for(i = 0; i < nb; i++)
        {
            MALLOCV(node->mu[i], float, ns);
            MALLOCV(node->sigma[i], float, ns);
        }
    }

    MALLOCV( node->delta, float, ns);
    MALLOCV( node->belief, float, nb );
    MALLOCV( node->beliefEuc, float, nb );
    MALLOCV( node->beliefMal, float, nb );
    MALLOCV( node->outputBelief, float, nb);
    MALLOCV( node->observation, float, ns );

    node->parent = NULL;
    if (layer == 0)
    {
        node->children = NULL;
    } else {
        MALLOC( node->children, Node *, childNumber );
        for ( i = 0; i < childNumber; i++ )
        {
            node->children[i] = NULL;
        }
    }

    // copy the input offset for the inputs (should be NULL for non-input nodes)
    if( inputOffsets == NULL ){
        node->inputOffsets = NULL;
    }else{
        MALLOC(node->inputOffsets, uint, ni);
        memcpy(node->inputOffsets, inputOffsets, sizeof(uint) * ni);
    }

    for( i=0; i < nb; i++ )
    {
        // init belief (node output)
        node->belief[i] = 1 / (float)nb;
        node->beliefEuc[i] = 1 / (float)nb;
        node->beliefMal[i] = 1 / (float)nb;
        node->outputBelief[i] = node->belief[i];

        if(!d->isUniform){
            node->nCounts[i] = 0;
            // init starv trace to one
            node->starv[i] = 1.0f;

            // init mu and sigma
            for(j=0; j < ns; j++)
            {
                node->mu[i][j] = (float) rand() / (float) RAND_MAX;
                node->sigma[i][j] = INIT_SIGMA;
            }
        }
    }

    //evenInitForMu(node->mu, nb, ns);

    for( i=0; i < ns; i++ )
    {
        node->observation[i] = (float) rand() / (float) RAND_MAX;
    }
}

// 2013.6.21
// CZT
// Uniformly initialize the centroids:
void evenInitForMu(float ** tempMu, int tempNb, int tempNs)
{
    int i,j;
    float aPiece = 1.0/tempNb;
    for(i=0; i<tempNb; ++i)
    {
        for(j=0; j<tempNs; ++j)
        {
            tempMu[i][j] = aPiece * i;
        }
    }
}

// deallocate the node.
void DestroyNode( Node *n)
{
    uint i;

    if(!n->d->isUniform){
        for (i = 0; i < n->nb; i++)
        {
            FREE( n->mu[i] );
            FREE( n->sigma[i] );
        }
        FREE( n->mu );
        FREE( n->sigma );
        FREE( n->nCounts );
        FREE( n->starv );
    }

    FREE( n->delta );
    FREE( n->belief );
    FREE( n->beliefEuc );
    FREE( n->beliefMal );
    FREE( n->outputBelief );
    FREE( n->observation );

    if( n->children != NULL )
    {
        FREE( n->children );
    }

    n->children = NULL;
    n->parent = NULL;

    // if it is a zero-layer node, free the input offset array on the host
    if( n->inputOffsets != NULL)
    {
        FREE(n->inputOffsets);
    }

    // 2013.4.16
    // CZT
    //
    n->belief = NULL;
    n->beliefEuc = NULL;
    n->beliefMal = NULL;
    n->outputBelief = NULL;
    n->observation = NULL;
    n->delta = NULL;
    n->inputOffsets = NULL;/**/
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

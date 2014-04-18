#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

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
#define SERIALIZE_VERSION 10

static void _initializeDestinParameters(uint *nb, uint *layerMaxNb, uint *layerWidths, uint inputDim, bool isUniform, uint extRatio, uint nl, uint nMovements,
                                Destin* d, uint nc, float *temp, float freqCoeff, float freqTreshold, float addCoeff);

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
    
    uint nl, i, nMovements, nc;

    // parse config file

    // get number of movements per digit
    fscanfResult = fscanf(configFile, "%d", &nMovements);
    printf("nMovements: %d\n", nMovements);

    // get number of distinct classes
    fscanfResult = fscanf(configFile, "%d", &nc);
    printf("# classes: %d\n", nc);

    // get number of layers
    fscanfResult = fscanf(configFile, "%d", &nl);

    //TODO: fix for layerMaxNb
    DestinConfig * dc = CreateDefaultConfig(nl);
    dc->nMovements = nMovements;
    dc->nClasses = nc;

    printf("# layers: %d\n", nl);
    // get layer beliefs and temps
    for(i=0; i < nl; i++) {
        fscanfResult = fscanf(configFile, "%u %u %u %f", &dc->layerWidths[i], &dc->centroids[i], &dc->layerMaxNb[i], &dc->temperatures[i]);
        printf("\t%u %u %u %f\n", dc->layerWidths[i], dc->centroids[i], dc->layerMaxNb[i], dc->temperatures[i]);
    }

    //TODO: update CONFIG.txt
    // get coeffs
    fscanfResult = fscanf(configFile, "%u", &dc->inputDim);
    fscanfResult = fscanf(configFile, "%f", &dc->beta);
    fscanfResult = fscanf(configFile, "%f", &dc->lambdaCoeff);
    fscanfResult = fscanf(configFile, "%f", &dc->gamma);
    fscanfResult = fscanf(configFile, "%f", &dc->starvCoeff);
    fscanfResult = fscanf(configFile, "%f", &dc->freqCoeff);
    fscanfResult = fscanf(configFile, "%f", &dc->freqTreshold);
    fscanfResult = fscanf(configFile, "%f", &dc->addCoeff);

    uint isRecurrent; // if clusters on previous beliefs and first parent's previous beliefs
    fscanfResult = fscanf(configFile, "%u", &isRecurrent);
    dc->isRecurrent = isRecurrent == 0 ? false : true;

    // is uniform, i.e. shared centroids
    // 0 = uniform off, 1 = uniform on
    uint iu;
    fscanfResult = fscanf(configFile, "%u", &iu);
    dc->isUniform = iu == 0 ? false : true;

    //TODO: set learning strat
    //TODO: fix test config file
    // applies boltzman distibution
    // 0 = off, 1 = on
    char bts[80]; //belief transform string

    fscanfResult = fscanf(configFile, "%80s", bts);

    BeliefTransformEnum bte = BeliefTransform_S_to_E(bts);

    printf("beta: %0.2f. lambda: %0.2f. gamma: %0.2f. starvCoeff: %0.2f\n",dc->beta, dc->lambdaCoeff, dc->gamma, dc->starvCoeff);
    printf("freqCoeff: %0.3f. freqTreshold: %0.3f. addCoeff: %0.3f ", dc->freqCoeff, dc->freqTreshold, dc->addCoeff);
    printf("isRecurrent: %s. isUniform: %s. belief transform: %s.\n", dc->isRecurrent ? "YES" : "NO", dc->isUniform ? "YES" : "NO", bts);

    newDestin = InitDestinWithConfig(dc);

    SetBeliefTransform(newDestin, bte);

    fclose(configFile);

    DestroyConfig(dc);
    return newDestin;
}

/** Calculates input offsets for square shape nodes from layer 0
 * @param layerWidth   - width of layer 0
 * @param nIdx - node index in the layer 0
 * @param ni   - dimensionality of input, should be square number
 * @param inputOffsets - pointer to preallocated uint array which will hold calculated offsets.
 */
static void CalcSquareNodeInputOffsets(uint layerWidth, uint nIdx, uint ni, uint * inputOffsets)
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
}

// Infers nci ( # children inputs ) from layer widths.
// returns true unless it recieves an unsupported heirarcy.
bool _InferNCIFromWidths(uint * layerWidths, uint * nci, uint nLayers){
    uint l;
    for(l = 1 ; l < nLayers ; l++){
        uint clw = layerWidths[l - 1]; // child layer width
        uint plw = layerWidths[l];     // parent layer width
        if(clw % plw == 0){ // divides evenly, assume non overlapping
            uint ratio = clw / plw;
            nci[l] = ratio * ratio;
        } else if(clw - plw == 1){ // assume overlapping and assume 4 to 1
            nci[l] = 4;
        } else {
            fprintf(stderr, "Non supported heirarchy from layer %i width %i to layer %i width %i!\n", l-1, clw, l, plw);
            fprintf(stderr, "Returning NULL Destin!\n");
            return false;
        }
    }
    return true;
}

Destin * InitDestinWithConfig(DestinConfig * config){
    return InitDestin(config->inputDim,
                      config->nLayers,
                      config->centroids,
                      config->layerMaxNb,
                      config->layerWidths,
                      config->nClasses,
                      config->beta,
                      config->lambdaCoeff,
                      config->gamma,
                      config->temperatures,
                      config->starvCoeff,
                      config->freqCoeff,
                      config->freqTreshold,
                      config->addCoeff,
                      config->nMovements,
                      config->isUniform,
                      config->extRatio,
                      config->isRecurrent
                      );
}

Destin * InitDestin(    // initialize Destin.
    uint inputDim,      // length of input vector for each bottom layer node. i.e. is 16 for 4x4 pixel input.
                        // numbers of children should be square
    uint nLayers,       // number of layers
    uint *nb,           // initial number of centroids for each layer
    uint *layerMaxNb,   // maximum number of centroids for each layer
    uint *layerWidths,  // width of each layer
    uint nc,            // number of classes
    float beta,         // beta coeff
    float lambdaCoeff,  // lambdaCoeff coeff
    float gamma,        // gamma coeff
    float *temp,        // temperature for each layer
    float starvCoeff,   // starv coeff
    float freqCoeff,    // frequency coeff
    float freqTreshold, // frequency treshold
    float addCoeff,     // TODO: comment
    uint nMovements,    // number of movements per digit presentation
    bool isUniform,     // is uniform - if nodes in a layer share one list of centroids
    uint extRatio,      // input extension ratio
    bool isRecurrent    // If nodes cluster on their firstParent's beliefs and their own previous beliefs
){
    uint i, l, maxNb, maxNs;
    Destin *d;

    // initialize a new Destin object
    MALLOC(d, Destin, 1);

    _initializeDestinParameters(nb, layerMaxNb, layerWidths, inputDim, isUniform, extRatio, nLayers, nMovements,
                                d, nc, temp, freqCoeff, freqTreshold, addCoeff);

    d->isRecurrent = isRecurrent;

    // keep track of the max num of beliefs and states.  we need this information
    // to correctly call kernels later
    maxNb = 0;
    maxNs = 0;

    uint n = 0;

    // initialize the rest of the network
    for( l=0; l < nLayers; l++ )
    {
        // update max belief
        if( nb[l] > maxNb )
        {
            maxNb = nb[l];
        }

        uint np = ((l + 1 == nLayers) ? 0 : nb[l+1]);
        uint ni = (l == 0 ? d->nci[0] : d->nci[l] * nb[l-1]);
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
                        lambdaCoeff,
                        temp[l],
                        &d->nodes[n],
                        (l > 0 ? NULL : inputOffsets),
                        (l > 0 ? d->nci[l] : 0)
                    );
        }//next node
    }//next layer

    if(!LinkParentsToChildren( d )){
        fprintf(stderr, "Could not link parent to children. Returning NULL destin structure!\n");
        return NULL;
    }

    d->maxNb = maxNb;
    d->maxNs = maxNs;

    return d;
}

static void _initializeDestinParameters(uint *nb, uint *layerMaxNb, uint *layerWidths, uint inputDim, bool isUniform, uint extRatio, uint nl, uint nMovements,
                                Destin* d, uint nc, float *temp, float freqCoeff, float freqTreshold, float addCoeff)
{
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
    d->addCoeff = addCoeff;

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

    MALLOC(d->layerMaxNb, uint, nl);
    memcpy(d->layerMaxNb, layerMaxNb, sizeof(uint)*nl);

    MALLOC(d->layerSize, uint, nl);
    MALLOC(d->layerNodeOffsets, uint, nl);
    MALLOC(d->layerWidth, uint, nl);
    memcpy(d->layerWidth, layerWidths, sizeof(uint) * nl);


    // init the train mask (determines which layers should be training)
    MALLOC(d->layerMask, uint, d->nLayers);
    for( i=0; i < d->nLayers; i++ )
    {
        d->layerMask[i] = 0;
    }

    for( i = 0 ; i < nl; i++){
        d->layerSize[i] = d->layerWidth[i] * d->layerWidth[i];
    }

    MALLOC(d->nci, uint, nl);
    d->nci[0] = inputDim;
    _InferNCIFromWidths(layerWidths, d->nci, nl);

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

void _LinkNonOverlapping(uint l, Destin *d)
{
    uint pr, pc;
    uint cr, cc;
    uint piw = (uint) sqrt(d->nci[l+1]);
    Node* parent_node;
    Node* child_node;
    uint child_layer_width = d->layerWidth[l];
    for(cr = 0 ; cr < child_layer_width; cr++){
        pr = cr / piw;
        for(cc = 0; cc < child_layer_width ; cc++){
            pc = cc / piw;
            child_node = GetNodeFromDestin(d, l, cr, cc);
            parent_node = GetNodeFromDestin(d, l+1, pr, pc);
            parent_node->children[(cr % piw) * piw + cc % piw] = child_node;
            child_node->parents[0] = parent_node;
            child_node->firstParent = parent_node;
            child_node->nParents = 1;
        }
    }
}

void _LinkOverlapping(uint l, Destin *d)
{
    Node* parent_node;
    Node* child_node;
    uint cr, cc, i;
    uint child_layer_width = d->layerWidth[l];
    for(cr = 0 ; cr < child_layer_width; cr++){
        for(cc = 0; cc < child_layer_width ; cc++){
            child_node = GetNodeFromDestin(d, l, cr, cc);
            if(cc > 0 && cr > 0){ // has north west parent
                parent_node = GetNodeFromDestin(d, l+1, cr-1, cc-1);
                child_node->parents[0] = parent_node;
                parent_node->children[3] = child_node;
                child_node->nParents++;
            }

            if(cc < child_layer_width - 1 && cr > 0){ // has nort east parent
                parent_node = GetNodeFromDestin(d, l+1, cr-1, cc);
                child_node->parents[1] = parent_node;
                parent_node->children[2] = child_node;
                child_node->nParents++;
            }

            if(cc > 0 && cr < child_layer_width - 1){ // has south west parent
                parent_node = GetNodeFromDestin(d, l+1, cr, cc-1);
                child_node->parents[2] = parent_node;
                parent_node->children[1] = child_node;
                child_node->nParents++;
            }

            if(cc < child_layer_width - 1 && cr < child_layer_width - 1){ // has south east parent
                parent_node = GetNodeFromDestin(d, l+1, cr, cc);
                child_node->parents[3] = parent_node;
                parent_node->children[0] = child_node;
                child_node->nParents++;
            }

            // set the first parent
            child_node->firstParent = NULL;
            for(i = 0 ; i < 4 ; i++){
                if(child_node->parents[i] != NULL){
                    child_node->firstParent = child_node->parents[i];
                    break;
                }
            }
        }
    }
}

/* The destin hierarchy is inferred from the destin->layerWidth array.

If a parent layer width is one less than the child layer, then it
is inferred that the child layer has overlapping nodes. In which
case the nodes can each have up to 4 parent nodes.

Otherwise, the child layer width must divide evenly by the parent
layer width i.e. childWidth % parentWidth == 0 and assumed that the
child nodes are non overlapping and each only have one parent.

Returns true if the layers were valid
*/
bool LinkParentsToChildren( Destin *d )
{
    uint l;

    for(l = 0 ; l + 1 < d->nLayers ; l++){
        int childLayerWidth = d->layerWidth[l];
        int parentLayerWidth = d->layerWidth[l+1];
        if(childLayerWidth % parentLayerWidth == 0) {
            // non overlappings
            // Child can be divided evenly
            _LinkNonOverlapping(l, d);
        } else if(childLayerWidth - parentLayerWidth == 1){
            _LinkOverlapping(l, d);
        } else {
            fprintf(stderr, "Unsupported structure from layer %i to %i.\n", l, l+1);
            return false;
        }
    }
    return true;
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
    FREE(d->layerMaxNb);
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


void _DstReadCheck(void * dest, size_t element_size, size_t count, FILE * file, int line){
    int read_count = fread(dest, element_size, count, file);
    if(read_count != count){
        int error = ferror(file);
        if(error){
            oops("File read error. Error number %i, line %i, file %s\n", error, line, __FILE__);
        } else if(feof(file)){
            oops("End of file reached, line %i, file %s\n", line, __FILE__);
        } else {
            oops("Unknown file read error\n");
        }
    }
    return;
}

// reads from the file and checks for errors
#define DstReadCheck(dest, size, count, file) { \
    _DstReadCheck(dest, size, count, file, __LINE__);\
}

void _DstWriteCheck(void * dest, size_t element_size, size_t count, FILE * file, int line){
    int write_count = fwrite(dest, element_size, count, file);
    if(write_count != count){
        int error = ferror(file);
        if(error){
            oops("File write error. Error number %i, line %i, file %s\n", error, line, __FILE__);
        } else if(feof(file)){
            oops("End of file reached, line %i, file %s\n", line, __FILE__);
        } else {
            oops("Unknown file read error\n");
        }
    }
    return;
}

// writes to the file and checks for errors
#define DstWriteCheck(dest, size, count, file) { \
    _DstWriteCheck(dest, size, count, file, __LINE__);\
}

void SaveDestin( Destin *d, char *filename )
{
    uint i, j, l;
    Node *nTmp;

    FILE *dFile;

    dFile = fopen(filename, "wb");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return;
    }

    DstWriteCheck(&d->serializeVersion, sizeof(uint), 1,       dFile);

    // write destin hierarchy information to disk
    DstWriteCheck(&d->nMovements,  sizeof(uint), 1,            dFile);
    DstWriteCheck(&d->nc,          sizeof(uint), 1,            dFile);
    DstWriteCheck(&d->nLayers,     sizeof(uint), 1,            dFile);
    DstWriteCheck(&d->isUniform,   sizeof(bool), 1,            dFile);
    DstWriteCheck(d->nb,           sizeof(uint), d->nLayers,   dFile);
    DstWriteCheck(d->layerMaxNb,   sizeof(uint), d->nLayers,   dFile);
    DstWriteCheck(d->layerWidth,   sizeof(uint), d->nLayers,   dFile);

    // write destin params to disk
    DstWriteCheck(d->temp,                 sizeof(float),              d->nLayers,  dFile);
    DstWriteCheck(&d->nodes[0].beta,       sizeof(float),              1,           dFile); //TODO consider moving these constants to the destin struc
    DstWriteCheck(&d->nodes[0].lambdaCoeff,sizeof(float),              1,           dFile);
    DstWriteCheck(&d->nodes[0].gamma,      sizeof(float),              1,           dFile);
    DstWriteCheck(&d->nodes[0].starvCoeff, sizeof(float),              1,           dFile);
    DstWriteCheck(&d->freqCoeff,           sizeof(float),              1,           dFile);
    DstWriteCheck(&d->freqTreshold,        sizeof(float),              1,           dFile);
    DstWriteCheck(&d->addCoeff,            sizeof(float),              1,           dFile);
    DstWriteCheck(&d->extRatio,            sizeof(uint),               1,           dFile);
    DstWriteCheck(&d->isRecurrent,         sizeof(bool),               1,           dFile);

    // save input dimensionality i.e. nci[0]
    DstWriteCheck(&d->nci[0],              sizeof(uint),               1,           dFile);

    DstWriteCheck(&d->centLearnStrat,      sizeof(CentroidLearnStrat), 1,           dFile);
    DstWriteCheck(&d->beliefTransform,     sizeof(BeliefTransformEnum),1,           dFile);
    DstWriteCheck(&d->fixedLearnRate,      sizeof(float),              1,           dFile);

    // write node belief states
    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];
        DstWriteCheck(nTmp->belief,           sizeof(float), nTmp->nb, dFile);
        DstWriteCheck(nTmp->outputBelief,     sizeof(float), nTmp->nb, dFile);
    }

    // write node statistics to disk
    if(d->isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0); //get the 0th node of the layer
            for (i = 0; i < d->nb[l]; i++){
                DstWriteCheck(d->uf_mu[l][i],          sizeof(float),  nTmp->ns,    dFile);
                DstWriteCheck(d->uf_sigma[l][i],       sizeof(float),  nTmp->ns,    dFile);
                DstWriteCheck(d->uf_avgDelta[l][i],    sizeof(float),  nTmp->ns,    dFile);
            }
            DstWriteCheck(d->uf_absvar[l],             sizeof(float),  nTmp->ns,               dFile);
            DstWriteCheck(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],               dFile);
            DstWriteCheck(d->uf_winFreqs[l],           sizeof(float),  d->nb[l],               dFile);
            DstWriteCheck(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],               dFile);
            DstWriteCheck(d->uf_persistWinCounts_detailed[l],   sizeof(long),   d->nb[l],      dFile);
            DstWriteCheck(d->uf_starv[l],              sizeof(float),  d->nb[l],               dFile);
        }
    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];
            // write statistics
            for (j = 0; j < nTmp->nb; j++){
                DstWriteCheck(nTmp->mu[j],          sizeof(float),  nTmp->ns,    dFile);
                DstWriteCheck(nTmp->sigma[j],       sizeof(float),  nTmp->ns,    dFile);
            }
            DstWriteCheck(nTmp->starv,     sizeof(float),  nTmp->nb,           dFile);
            DstWriteCheck(nTmp->nCounts,   sizeof(long),   nTmp->nb,           dFile);
        }
    }

    fclose(dFile);
}

Destin * LoadDestin( Destin *d, const char *filename )
{
    FILE *dFile;
    uint i, j, l;
    Node *nTmp;

    if( d != NULL )
    {
        fprintf(stderr, "Pointer 0x%p is already initialized, clearing!!\n", d);
        DestroyDestin( d );
    }

    uint serializeVersion;
    uint nMovements, nc, nl;
    bool isUniform, isRecurrent;
    uint extendRatio;
    uint *nb, *layerMaxNb, *layerWidths;

    float beta, lambdaCoeff, gamma, starvCoeff, freqCoeff, freqTreshold, addCoeff;
    float *temp;

    dFile = fopen(filename, "rb");
    if( !dFile )
    {
        fprintf(stderr, "Error: Cannot open %s", filename);
        return NULL;
    }

    DstReadCheck(&serializeVersion, sizeof(uint), 1, dFile);
    if(serializeVersion != SERIALIZE_VERSION){
        fprintf(stderr, "Error: can't load %s because its version is %i, and we're expecting %i\n", filename, serializeVersion, SERIALIZE_VERSION);
        return NULL;
    }

    // read destin hierarchy information from disk
    DstReadCheck(&nMovements,  sizeof(uint), 1, dFile);
    DstReadCheck(&nc,          sizeof(uint), 1, dFile);
    DstReadCheck(&nl,          sizeof(uint), 1, dFile);
    DstReadCheck(&isUniform,   sizeof(bool), 1, dFile);

    MALLOC(nb, uint, nl);
    MALLOC(layerMaxNb, uint, nl);
    MALLOC(layerWidths, uint, nl);
    MALLOC(temp, float, nl);

    DstReadCheck(nb,            sizeof(uint),    nl,   dFile);
    DstReadCheck(layerMaxNb,    sizeof(uint),    nl,   dFile);
    DstReadCheck(layerWidths,   sizeof(uint),    nl,   dFile);

    // read destin params from disk
    DstReadCheck(temp,          sizeof(float),    nl,  dFile);
    DstReadCheck(&beta,         sizeof(float),    1,   dFile);
    DstReadCheck(&lambdaCoeff,  sizeof(float),    1,   dFile);
    DstReadCheck(&gamma,        sizeof(float),    1,   dFile);
    DstReadCheck(&starvCoeff,   sizeof(float),    1,   dFile);
    DstReadCheck(&freqCoeff,    sizeof(float),    1,   dFile);
    DstReadCheck(&freqTreshold, sizeof(float),    1,   dFile);
    DstReadCheck(&addCoeff,     sizeof(float),    1,   dFile);
    DstReadCheck(&extendRatio,  sizeof(uint),     1,   dFile);
    DstReadCheck(&isRecurrent,  sizeof(bool),     1,   dFile);

    //TODO: verify the results of the read

    uint inputDim; // i.e. destin->nci[0]
    DstReadCheck(&inputDim,     sizeof(uint),     1,   dFile);

    d = InitDestin(inputDim, nl, nb, layerMaxNb, layerWidths, nc, beta, lambdaCoeff, gamma, temp, starvCoeff,
                   freqCoeff, freqTreshold, addCoeff, nMovements, isUniform, extendRatio, isRecurrent);

    // temporary arrays were copied in InitDestin
    FREE(nb); nb = NULL;
    FREE(layerMaxNb); layerMaxNb = NULL;
    FREE(temp); temp = NULL;
    FREE(layerWidths); layerWidths = NULL;


    DstReadCheck(&d->centLearnStrat,sizeof(CentroidLearnStrat),   1,         dFile);
    SetLearningStrat(d, d->centLearnStrat);

    DstReadCheck(&d->beliefTransform, sizeof(BeliefTransformEnum),1,         dFile);
    SetBeliefTransform(d, d->beliefTransform);

    DstReadCheck(&d->fixedLearnRate,sizeof(float),                1,         dFile);

    // load node belief states
    for( i=0; i < d->nNodes; i++ )
    {
        nTmp = &d->nodes[i];
        DstReadCheck(nTmp->belief,       sizeof(float), nTmp->nb, dFile);
        DstReadCheck(nTmp->outputBelief, sizeof(float), nTmp->nb, dFile);
    }

    if(isUniform){
        for(l = 0 ; l < d->nLayers; l++){
            nTmp = GetNodeFromDestin(d, l, 0, 0);
            for(i = 0 ; i < d->nb[l]; i++){
                DstReadCheck(d->uf_mu[l][i],          sizeof(float),  nTmp->ns,    dFile);
                DstReadCheck(d->uf_sigma[l][i],       sizeof(float),  nTmp->ns,    dFile);
                DstReadCheck(d->uf_avgDelta[l][i],    sizeof(float),  nTmp->ns,    dFile);
            }
            DstReadCheck(d->uf_absvar[l],             sizeof(float),  nTmp->ns,    dFile);
            DstReadCheck(d->uf_winCounts[l],          sizeof(uint),   d->nb[l],    dFile);
            DstReadCheck(d->uf_winFreqs[l],           sizeof(float),  d->nb[l],    dFile);
            DstReadCheck(d->uf_persistWinCounts[l],   sizeof(long),   d->nb[l],    dFile);
            DstReadCheck(d->uf_persistWinCounts_detailed[l],   sizeof(long),   d->nb[l],    dFile);
            DstReadCheck(d->uf_starv[l],              sizeof(float),  d->nb[l],    dFile);
        }

    }else{
        for( i=0; i < d->nNodes; i++ )
        {
            nTmp = &d->nodes[i];

            // load statistics
            for(j = 0 ; j < nTmp->nb; j++){
                DstReadCheck(nTmp->mu[j],    sizeof(float),  nTmp->ns,    dFile);
                DstReadCheck(nTmp->sigma[j], sizeof(float),  nTmp->ns,    dFile);
            }
            DstReadCheck(nTmp->starv, sizeof(float), nTmp->nb, dFile);
            DstReadCheck(nTmp->nCounts, sizeof(long), nTmp->nb, dFile);
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
    float       lambdaCoeff,
    float       temp,
    Node        *node,
    uint        *inputOffsets,
    uint        nChildren
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
    node->lambdaCoeff   = lambdaCoeff;
    node->gamma         = gamma;
    node->temp          = temp;
    node->winner        = 0;
    node->layer         = layer;
    node->nChildren     = nChildren;
    node->nParents      = 0; // will be updated in LinkParentsToChildren()

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

    MALLOCV( node->delta, float, ns );
    MALLOCV( node->belief, float, nb );
    MALLOCV( node->beliefEuc, float, nb );
    MALLOCV( node->beliefMal, float, nb );
    MALLOCV( node->outputBelief, float, nb );
    MALLOCV( node->observation, float, ns );

    if (layer == 0){
        node->children = NULL;
    } else {
        MALLOC( node->children, Node *, nChildren );
        for ( i = 0; i < nChildren; i++ )
        {
            node->children[i] = NULL;
        }
    }

    // Assume each node may have up to 4 parents
    // in case of overlapping regions.
    MALLOC( node->parents, Node *, 4);
    for( i = 0 ; i < 4; i++)
    {
        node->parents[i] = NULL;
    }

    node->firstParent = NULL;

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

    if( n->parents != NULL )
    {
        FREE( n->parents );
    }

    n->parents = NULL;

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


DestinConfig* CreateDefaultConfig(uint layers){
    // Defaults to classic 4 to 1 non overlapping heirarcy.

    DestinConfig * config;
    MALLOC(config, DestinConfig, 1);
    config->addCoeff = 0;
    config->beta = 0.001;

    uint i, defaultCentroidsPerLayer = 5;

    MALLOC(config->centroids, uint, layers);
    MALLOC(config->layerMaxNb, uint, layers);
    for(i = 0 ; i < layers ; i++){
        config->centroids[i] = defaultCentroidsPerLayer;
        config->layerMaxNb[i] = defaultCentroidsPerLayer;
    }

    config->extRatio = 1;
    config->freqCoeff =  0.05;
    config->freqTreshold = 0;
    config->gamma = 0.10;
    config->inputDim = 16;  // 4x4 pixel input per bottom layer node
    config->isRecurrent = false;
    config->isUniform = true;
    config->lambdaCoeff = 0.10;

    MALLOC(config->layerWidths, uint, layers);
    for(i = 0 ; i < layers ; i++){
        // 16, 8, 4, 2, 1... ect.
        config->layerWidths[i] = (uint)pow(2, layers - i - 1);
    }

    config->nClasses = 0;
    config->nLayers = layers;
    config->nMovements = 0;
    config->starvCoeff = 0.12;

    MALLOC(config->temperatures, float, layers);
    for(i = 0; i < layers; i++){
        config->temperatures[i] = config->centroids[i] * 2.0;
    }
    return config;
}

void DestroyConfig(DestinConfig * c){
    FREE(c->centroids);
    FREE(c->temperatures);
    FREE(c->layerWidths);
    FREE(c->layerMaxNb);
    FREE(c);
}

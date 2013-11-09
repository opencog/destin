#ifndef __DESTIN_H
#define __DESTIN_H

#include <stdbool.h>
#include "macros.h"
#include "node.h"
#include "learn_strats.h"
#include "belief_transform.h"

#define INIT_SIGMA 0.00001

typedef struct Destin {
    uint serializeVersion;              // Identifies the compaitibility version of this destin structure during saves and loads.
    uint maxNb;                         // max number of beliefs for all nodes (important for kernels)
    uint maxNs;
    uint nc;                            // number of classes to discriminate
    uint nNodes;                        // number of nodes in the entire destin network
    uint nMovements;                    // number of movements per digit presentation
    uint nLayers;                       // number of layers in network
    float muSumSqDiff;
    uint *nb;                           // current number of beliefs in a node of a layer
    uint *layerMaxNb;                   // max number of beliefs for layer (upper limit for centroid addition)
    uint *nci;                          // input dimensionality for layer 0 and number of children for layers above zero

    struct Node * nodes;                // pointer to list of host nodes

    float       * temp;                 // temperatures for each layer, used with the belief tranform functions.
    float       * dataSet;              // pointer to dataset

    uint        * inputLabel;           // input label (used during supervised training)
    uint        * layerSize;            // size for each layer ( nodes per layer )
    uint        * layerWidth;           // node width for each layer
    uint        * layerNodeOffsets;     // stores the node id of the first node of each layer
    uint        * layerMask;            // controls which layers are training. 1 = train, 0 = not train.

    CentroidLearnStrat   centLearnStrat;        // centroid learning strategy enum
    CentroidLearnStratFunc centLearnStratFunc;  // centroid learning strategy function pointer

    BeliefTransformEnum beliefTransform; // which belief transform to apply to output beliefs.
    BeliefTransformFunc beliefTransformFunc; // function pointer for the belief transform

    float       freqCoeff;              // coefficient for updating centroid's estimated frequency (d->winFreqs)
    float       freqTreshold;           // if centroid's estimated frequency deteriorate below the treshold the centroid is wiped out
    float       addCoeff;               // coefficient for estimating probablity of adding a new centroid to a layer
    float       fixedLearnRate;         // if CLS_Fixed is set for centLearnStrat, then this is the fixed learning rate to use, otherwise ignored.

    bool        isUniform;              // internal flag to determine if this destin has been made uniform
                                        // which means all nodes in a layer share their centroids

    // uniform destin shared centroid variables
    float       *** uf_mu;              // shared centroids location, resizable array of size nl x nb x ns
    float       *** uf_sigma;           // shared centroids sigma, resizable array of size nl x nb x ns
    float       ** uf_absvar;           // layers absolute deviation, resizable array of size nl x ns

    uint        ** uf_winCounts;        // Counts how many nodes in a layer pick the given centroid as winner in one call of ForumateBeliefs
    long        ** uf_persistWinCounts; // keeps track how many times the shared centroids win over the lifetime of the training the destin network.
    long        ** uf_persistWinCounts_detailed;  // Because uf_persistWinCounts just count once when a centroid won,
                                        // this counting array contains all counts for all node
    float       ** uf_winFreqs;         // Estimated frequency how many nodes in a layer pick the given centroid as winner
                                        // in recent history of calls of FormulateBelief (on-line algorithm)

    float       *** uf_avgDelta;        // used to average node centroid movement vectors
    float       *** uf_avgSquaredDelta;
    float       ** uf_avgAbsDelta;

    float       ** uf_starv;            // shared centroids starvation

    int         inputImageSize;         // d->layerSize[0] * d->nci[0]

    /* Extend the input size by this integer amount.
       Used for example to take in 2 images for stero vision for example,
       or 3 RGB images at once. */
    int         extRatio;

    bool        isRecurrent;            // If nodes cluster on their firstParent's beliefs and their own previous beliefs

} Destin; // end Destin typedef

typedef struct DestinConfig {
    float  addCoeff;
    float  beta;
    uint  *centroids;       //TODO: make naming consistent with nb
    int    extRatio;
    float  freqCoeff;
    float  freqTreshold;
    float  gamma;
    uint   inputDim;        // pixels input per bottom layer node
    bool   isRecurrent;
    bool   isUniform;
    float  lambdaCoeff;
    uint  *layerWidths;
    uint  *layerMaxNb;      //TODO: make consistent with centroids
    uint   nClasses;
    uint   nLayers;
    uint   nMovements;
    float  starvCoeff;
    float *temperatures;

} DestinConfig; // end DestinConfig typedef

/* Destin Functions Begin */
Destin * CreateDestin(                  // create destin from a config file
                    char *              // filename
        );

Destin * InitDestinWithConfig(DestinConfig *);

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
    float addCoeff,     // coefficient for estimating probablity of adding a new centroid to a layer
    uint nMovements,    // number of movements per digit presentation
    bool isUniform,     // is uniform - if nodes in a layer share one list of centroids
    uint extRatio,      // input extension ratio
    bool isRecurrent    // If nodes cluster on their firstParent's beliefs and their own previous beliefs
);


bool LinkParentsToChildren(             // link parents to their children
                    Destin *            // initialized destin pointer
                );

void TrainDestin(                       // train destin.
                 Destin *,              // initialized destin pointer
                 char *,                // filename for data file
                 char *                 // filename for label file
                );

void TestDestin(                        // test destin.
                 Destin *,              // initialized destin pointer
                 char *,                // filename for data file
                 char *,                // filename for label file
                 bool                   // generative/output ?
                );

void SaveDestin(                        // save destin to disk
                Destin *,               // network to save
                char *                  // filename to save to
        );

Destin * LoadDestin(                    // load destin from disk
                Destin *,               // network to save
                const char *            // filename to load from
        );

void ResetStarvTrace(                   // reset the starv trace to 1's
            Destin *
        );

void DestroyDestin(
                    Destin *
                  );

void FormulateBelief(                   // form belief operation.  gets the current belief from Destin
                    Destin *,           // network to obtain belief from
                    float *             // input
                );

void GenerateInputFromBelief(
                    Destin *,
                    float *
        );

void DisplayFeatures(
                    Destin *
        );

/**
 * same as DisplayFeatures but for other layer than 0
 */
void DisplayLayerFeatures(
                    Destin *d,
                    int layer,          // layer to show features for
                    int node_start,     // node start
                    int nodes           // number of nodes in the layer to show, if 0 then show them all
        );

void CopyOutputBeliefs(                 // copy previous beliefs into nodes output
                  Destin *              // pointer to destin object
                 );

void ClearBeliefs(                      // cleanse the pallette
                  Destin *              // pointer to destin object
                 );


// grab a node at a particular layer, row, and column
struct Node * GetNodeFromDestin(
                        Destin *d,      // pointer to destin object
                        uint l,         // layer
                        uint r,         // row
                        uint c          // column
                        );

// grab a node at a particular layer, and node index
struct Node * GetNodeFromDestinI(
                        Destin *d,      // pointer to destin object
                        uint l,         // layer
                        uint nIdx       // index of node in the layer. The first node each layer has index 0.
                        );

// Fills beliefs array in output beliefs for all nodes from given layer.
void GetLayerBeliefs(
                    Destin *d,          // pointer to destin object
                    uint layer,         // layer
                    float * beliefs     // output beliefs array. The caller should allocate the array.
                    );

//resets sharedCentroidsDidWin vector for each layer
void Uniform_ResetStats(
                            Destin *
                          );


DestinConfig* CreateDefaultConfig(uint layers);
void DestroyConfig(DestinConfig *);

/* Destin Functions End */
#endif

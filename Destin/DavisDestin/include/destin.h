#ifndef __DESTIN_H
#define __DESTIN_H

#include <stdbool.h>
#include "macros.h"
#include "node.h"
#include "learn_strats.h"
#include "belief_transform.h"

#define INIT_SIGMA 0.00001


/* Destin Struct Definition */
typedef struct Destin {
    uint serializeVersion;
    uint nInputPipeline;                // number of beliefs to copy to next nodes' input
    uint maxNb;                         // max number of beliefs for all nodes (important for kernels)
    uint maxNs;
    uint nc;                            // number of classes to discriminate
    uint nBeliefs;                      // number of beliefs ( sum over all layers of centroids per node * number of nodes per layer )
    uint nNodes;                        // number of nodes in the entire destin network
    uint nMovements;                    // number of movements per digit presentation
    uint nLayers;                       // number of layers in network
    float muSumSqDiff;
    uint *nb;                           // number of beliefs in a node of a layer
    
    struct Node * nodes;                // pointer to list of host nodes

    float       * belief;               // concatonated belief vector for all nodes
    float       * temp;                 // temperatures for each layer
    float       * dataSet;              // pointer to dataset
    float       * inputPipeline;        // concatonated input for all internal layer nodes

    uint        * inputLabel;           // input label (used during supervised training)
    uint        * layerSize;            // size for each layer ( nodes per layer )
    uint        * layerWidth;           // node width for each layer
    uint        * layerNodeOffsets;     // stores the node id of the first node of each layer
    uint        * layerMask;            // controls which layers are training. 1 = train, 0 = not train.

    CentroidLearnStrat   centLearnStrat;        // centroid learning strategy enum
    CentroidLearnStratFunc centLearnStratFunc;  // centroid learning strategy function pointer

    BeliefTransformEnum beliefTransform;
    BeliefTransformFunc beliefTransformFunc;

    float       fixedLearnRate;       // if CLS_Fixed is set for centLearnStrat, then this is the fixed learning rate to use, otherwise ignored.

    bool        isUniform;              // internal flag to determine if this destin has been made uniform
                                        // which means all nodes in a layer share their centroids

    //uniform destin shared centroid variables
    uint        ** uf_winCounts;        //counts how many nodes in a layer pick the given centroid as winner in one call of ForumateBeliefs
    float       ** uf_avgDelta;         //used to average node centroid movement vectors
    long        ** uf_persistWinCounts; //keeps track how many times the shared centroids win over the lifetime of the training the destin network.
    float       ** uf_sigma;            //shared centroids sigma, one array per layer of size nb x ns
    float       ** uf_starv;            //shared centroids starvation

    // 2013.4.11
    // CZT
    //
    int size;
    int extRatio;
    float ** uf_sigma_c1;
    float ** uf_avgDelta_c1;
} Destin  ;
/* Destin Struct Definition End */

/* Destin Functions Begin */
Destin * CreateDestin(                  // create destin from a config file
                    char *              // filename
        );

Destin * InitDestin(                    // initialize Destin.
                    uint,               // input dimensionality for first layer, input must be square
                    uint,               // number of layers
                    uint *,             // belief dimensionality for each layer
                    uint,               // number of classes
                    float,              // beta coeff
                    float,              // lambda coeff
                    float,              // gamma coeff
                    float *,            // temperature for each layer
                    float,              // starv coeff
                    uint,               // number of movements per digit presentation
                    bool               // is uniform - if nodes in a layer share one list of centroids
                );

// 2013.4.11
// CZT
//
Destin * InitDestin_c1(                    // initialize Destin.
    uint,               // input dimensionality for first layer, input must be square
    uint,               // number of layers
    uint *,             // belief dimensionality for each layer
    uint,               // number of classes
    float,              // beta coeff
    float,              // lambda coeff
    float,              // gamma coeff
    float *,            // temperature for each layer
    float,              // starv coeff
    uint,               // number of movements per digit presentation
    bool,               // is uniform - if nodes in a layer share one list of centroids
    int,
    int
);

void LinkParentBeliefToChildren(        // link the belief from a parent to the child for advice
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

// 2013.4.11
// CZT
//
void DestroyDestin_c1(
                    Destin *
                  );

void FormulateBelief(                   // form belief operation.  gets the current belief from Destin
                    Destin *,           // network to obtain belief from
                    float *             // input
                );

// 2013.4.11
// CZT
//
void FormulateBelief_c1(                   // form belief operation.  gets the current belief from Destin
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
                        uint nIdx       // node index
                        );

//resets sharedCentroidsDidWin vector for each layer
void Uniform_ResetStats(
                            Destin *
                          );





/* Destin Functions End */

#endif

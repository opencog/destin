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
    uint serializeVersion;              // Identifies the compaitibility version of this destin structure during saves and loads.s
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

    float       * belief;               // concatenated belief vector for all nodes
    float       * temp;                 // temperatures for each layer
    float       * dataSet;              // pointer to dataset

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
    uint        ** uf_winCounts;        // Counts how many nodes in a layer pick the given centroid as winner in one call of ForumateBeliefs
                                        //
    float       ** uf_avgDelta;         //used to average node centroid movement vectors
    long        ** uf_persistWinCounts; //keeps track how many times the shared centroids win over the lifetime of the training the destin network.
    float       ** uf_sigma;            //shared centroids sigma, one array per layer of size nb x ns
    float       ** uf_starv;            //shared centroids starvation

    /*The following is coded by CZT*/
    // 2013.6.10
    int sizeInd;
    int * nearInd;
    //2013.7.2
    int inputImageSize;
    int extRatio;
    // 2013.6.13, 2013.7.3
    long ** uf_persistWinCounts_detailed;  // Because uf_persistWinCounts just count once when a centroid won,
                                           // I think one more counting array should be necessary;
    float ** uf_absvar;
    float ** uf_avgSquaredDelta;
    float ** uf_avgAbsDelta;
    /*END*/
} Destin  ;
/* Destin Struct Definition End */

/* Destin Functions Begin */
Destin * CreateDestin(                  // create destin from a config file
                    char *              // filename
        );

Destin * InitDestin(    // initialize Destin.
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
    int                 // extRatio
);

// 2013.5.31
void addCentroid(Destin *d, uint ni, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma,
                  float *temp, float starvCoeff, uint nMovements, bool isUniform, int extRatio,
                  int currLayer, float **sharedCen, float **starv, float **sigma,
                  long ** persistWinCounts, long ** persistWinCounts_detailed, float ** absvar);
// 2013.6.6
void killCentroid(Destin *d, uint ni, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma,
                  float *temp, float starvCoeff, uint nMovements, bool isUniform, int extRatio,
                  int currLayer, int kill_ind, float **sharedCen, float **starv, float **sigma,
                  long **persistWinCounts, long ** persistWinCounts_detailed, float ** absvar);

void LinkParentsToChildren(             // link parents to their children
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

/* Destin Functions End */

#endif

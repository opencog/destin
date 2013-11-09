#ifndef __NODE_H
#define __NODE_H

#include "macros.h"

#define EPSILON     1e-8
#define MAX_INTERMEDIATE_BELIEF (1.0 / EPSILON)

#define USE_MAL     // use mahalanobis distance to calulate beliefs ( gives better results )
//#define USE_EUC   // use euclidian distance to calculate beliefs

#if defined(USE_MAL) && defined(USE_EUC)
#error "both USE_MAL and USE_EUC can't be set"
#endif

//#define STARV_QUICK_RESET  // if a centroid wins its starvation is immedialy reset to 1.0
#define STARV_SLOW_RESET     // if a centroid wins its starvation get increasead by a constant ( gives better results )

/* Node Struct Definition */
typedef struct Node {
    /* HOST VARIABLES BEGIN */
    struct Destin * d;      // referece to parent destin network
    // node parameters
    uint    nIdx;           // node id. Can retrieve this node from a destin heirarchy like this: destin->nodes[nIdx]
    uint     nb;            // number of beliefs ( number of centroids )
    uint     ni;            // number of inputs ( dimensionality of input vector)
    uint     ns;            // number of states ( dimensionality of centroids) = ni+nb+np+nc
                            // Centroids cluster on input, its own previous belief, parent's belief and "class" vector ( not implemented)

    uint     np;            // number of beliefs (centroids) for parent
    uint     nc;            // number of classes
    float   starvCoeff;     // starvation coefficient
    float   beta;           // sigma update weight (centroid variance)
    float   lambdaCoeff;    // parent previous belief weighting
    float   gamma;          // previous belief weighting
    float   temp;           // temperature for boltzmann normalization

    uint     winner;        // winning centroid index
    uint     genWinner;     // winning centroid index for generative procedure

    // node statistics
    float ** mu;            // centroid locations (resizable array nb x ns)
                            // in uniform destin, all nodes in a layer share the pointer to d->uf_mu[layer]
    float ** sigma;         // centroid variances (resizable array nb x ns)
    float * starv;          // centroid starvation coefficients. ( points to destin->uf_starv if it's uniform destin)

    float   muSqDiff;
    long  * nCounts;        // number of observation counts. How many times each centroid is picked as winner over all iterations.

    uint  * inputOffsets;   // offsets for each pixel taken from framePtr for this node. vector of length ni.
                            // (null for non-input layer nodes)
    float * observation;    // contains the node's input, previous 
                            // belief, and parent's previous belief ( length ni+nb+np )
    
    // node beliefs
    float * beliefEuc;      // belief (euclidean distance), length nb
    float * beliefMal;      // belief (malhanobis distance)
    float * belief;         // previous belief (euclidean or mal)
    float * outputBelief;   // output belief is used as parent node observation (input from child)

    struct Node ** parents; // Array of size 4 of pointers to parent nodes.
                            // 0 = NW (north west) parent, 1 = NE, 2 = SW, 3 = SE
                            // Element of the array will be null if it does not have a parent in the corresponding location.
                            // If all nodes in the node's layer have only 1 parent (for non overlapping node regions),
                            // then the parent is always placed in the 0th element and the rest are null.

    struct Node * firstParent; // First parent node that is not NULL.

    uint nParents;          // Number of parent nodes. There may be more than one when using overlapping node regions.

    struct Node ** children;// array of nChildren child node pointers (only for layers above 0)
    uint nChildren;         // number of children

    float * delta;          // vector that stores difference between observation and mu shared centroid vector
    uint    layer;          // layer this node belongs in
    uint    row;            // row of the layer this node belongs
    uint    col;            // column of the layer this node belongs
    /* HOST VARIABLES END */

} Node;

/* Node Functions Begin */
void  InitNode(                         // initialize a node.
                 uint,                  // node index
                 struct Destin *,       // reference to parent destin network
                 uint,                  // layer this node belongs to
                 uint,                  // belief dimensionality (# centroids)
                 uint,                  // input dimensionality (# input values)
                 uint,                  // parent belief dimensionality
                 uint,                  // number of classes
                 uint,                  // ns = state dimensionality (number of inputs + number of previous beliefs + number of parent's previous beliefs)
                                        // = ni + nb + np + nc
                 float,                 // starvation coefficient
                 float,                 // beta (sigma step size)
                 float,                 // lambda
                 float,                 // gamma
                 float,                 // temperature
                 Node *,                // pointer node on host
                 uint *,                // input offsets from input image (NULL for any non-input node)
                 uint                   // number of children
                );

// Recalculates ns, checks destin maxNb i MaxNs
void UpdateNodeSizes(
                 Node *,                // pointer to node
                 uint ni,               // new input dimensionality
                 uint nb,               // new previous belief dimensionality
                 uint np,               // new parent belief dimensionality
                 uint nc                // new context dimensionality
);

// 2013.6.21
void evenInitForMu(float ** tempMu, int tempNb, int tempNs);

void DestroyNode(
                 Node *
                );

void GetObservation(
                    Node *,             // pointer to list of nodes
                    float *,            // pointer to input frame
                    uint                // node index
                );

void CalculateDistances(
                    Node *,             // pointer to list of nodes
                    uint                // node index
                );

void NormalizeBeliefGetWinner(
                    Node *,             // pointer to list of nodes
                    uint                // node index
                );

void CalcCentroidMovement(
                    Node *,             // pointer to list of nodes
                    uint *,             // pointer to current class label
                    uint                // node index
                );

void MoveCentroids(
                    Node *,             // pointer to list of nodes
                    uint                // node index
                );

void UpdateStarvation(
                    Node *,             // pointer to list of nodes
                    uint                // node index
                );
 
void Uniform_AverageDeltas(
                    Node *,             // pointer to list of nodes
                    uint                // node index
                );

void Uniform_ApplyDeltas(
                    struct Destin *,
                    uint,               // layer to apply deltas
                    float **            // shared sigma float array to use. Table nb x ns
                );

void Uniform_UpdateStarvation(
                    struct Destin  *,
                    uint,               // layer
                    float * ,           // shared starvation vector for given layer
                    uint * ,            // shared centroids win counts vector for given layer
                    float               // starvation coefficient
                );

void Uniform_UpdateFrequency(
                    struct Destin  *,
                    uint,               // layer
                    float * ,           // shared estimated frequency vector for given layer
                    uint * ,            // shared centroids win counts vector for given layer
                    float               // estimated frequency coefficient
                );

void Uniform_DeleteCentroids(
                    struct Destin  *
                );

void Uniform_AddNewCentroids(
                    struct Destin  *
                );


#endif

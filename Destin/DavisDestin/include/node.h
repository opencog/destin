#ifndef __NODE_H
#define __NODE_H

#include "macros.h"

#define EPSILON     1e-8
#define MAX_INTERMEDIATE_BELIEF (1.0 / EPSILON)

// Use at 2013.6.5
//#define RECURRENCE_ON // if defined then it clusters on its previous and parent's previous belief.

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
	struct Destin *  d;			// referece to parent destin network
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
    float   nLambda;        // pBelief weighting
    float   gamma;          // parent pBelief weighting
    float   temp;           // temperature for boltzmann normalization

    uint     winner;        // winning centroid index
    uint     genWinner;     // winning centroid index for generative procedure

    // node statistics
    //TODO: make a uf_mu for uniform shared centroids instead of treating it differently
    float * mu;             // centroid locations ( a table nb x ns  ) . In uniform destin, all nodes in a layer share this pointer)
                            // See GetObservation for structure of centroids. Input -> own beliefs -> parent beliefs -> context ( unused )

    float * sigma;          // centroid variances
    float * starv;          // centroid starvation coefficients. ( points to destin->uf_starv if it's uniform destin)

    float   muSqDiff;
    long  * nCounts;        // number of observation counts. How many times each centroid is picked as winner over all iterations.

    uint  * inputOffsets;   // offsets for each pixel taken from framePtr for this node. vector of length ni.
                            // (null for non-input layer nodes)
    float * observation;    // contains the node's input, previous 
                            // belief, and parent's previous belief ( length ni+nb+np )
    float *genObservation;
    
    // node beliefs
    float * beliefEuc;      // belief (euclidean distance), length nb
    float * beliefMal;      // belief (malhanobis distance)
    float * pBelief;        // previous belief (euclidean or mal)
    float * outputBelief;   // output belief is used as parent node observation (input from child)

    struct Node * parent;   // pointer to parent node (null for to//p layer node)
    struct Node ** children;// array of 4 child node pointers

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
                 float *                // pointer to shared centroids for nodes in a layer. Is NULL if centroids are not shared ( i.e. classic destin, non uniform)
                );

// 2013.6.21
void evenInitForMu(float * tempMu, int tempNb, int tempNs);

// 2013.6.3, 2013.7.3
// CZT: updateCentroid_node for adding or killing;
void updateCentroid_node(
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
                 float *                // pointer to shared centroids for nodes in a layer. Is NULL if centroids are not shared ( i.e. classic destin, non uniform)
                );

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
                    float *             // shared sigma float array to use. Table nb x ns
                );

void Uniform_UpdateStarvation(
                    struct Destin  *,
                    uint,               // layer
                    float * ,           // shared starvation vector for given layer
                    uint * ,            // shared centroids win counts vector for given layer
                    float               // starvation coefficient
                );

#endif

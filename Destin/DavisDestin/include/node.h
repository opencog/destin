#ifndef __NODE_H
#define __NODE_H

#include "macros.h"


/* Node Struct Definition */
typedef struct Node {
    /* HOST VARIABLES BEGIN */
	struct Destin *  d;			// referece to parent destin network
    // node parameters
    uint    nIdx;           // node id
    uint     nb;            // number of beliefs ( number of centroids )
    uint     ni;            // number of inputs ( dimensionality of input vector)
    uint     ns;            // number of states ( dimensionality of centroids) = ni+nb+np+nc;
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
    float * sigma;          // centroid variances
    float * starv;          // centroid starvation coefficients. ( points to destin->uf_starv if it's uniform destin)

    float   muSqDiff;
    long  * nCounts;        // number of observation counts. How many times each centroid is picked as winner over all iterations.
    
    // node input
    float * input;          // input pointer (null for input layer nodes)
    uint  * inputOffsets;   // offsets for each pixel taken from framePtr for this node. vector of length ni.
                            // (null for non-input layer nodes)
    float * observation;    // contains the node's input, previous 
                            // belief, and parent's previous belief ( length ni+nb+np )
    float *genObservation;
    
    // node beliefs
    float * beliefEuc;      // belief (euclidean distance), length nb
    float * beliefMal;      // belief (malhanobis distance)
    float * pBelief;        // previous belief (euclidean)
    float * parent_pBelief; // parent previous belief

    struct Node ** children;// array of 4 child node pointers

    float * delta;           // vector that stores difference between observation and mu shared centroid vector
    uint    layer;          // layer this node belongs in
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
                 float *,               // pointer to input on host
                 float *,               // pointer to belief on host
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

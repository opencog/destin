#ifndef __NODE_H
#define __NODE_H

#include "macros.h"


/* Node Struct Definition */
typedef struct Node {
    /* HOST VARIABLES BEGIN */
	struct Destin *  d;			// referece to parent destin network
    // node parameters
    uint     nb;            // number of beliefs ( number of centroids )
    uint     ni;            // number of inputs ( dimensionality of input vector)
    uint     ns;            // number of states ( dimensionality of centroids) = ni+nb+np+nc;
    uint     np;            // number of beliefs (centroids) for parent
    uint     nc;            // number of classes
    float   starvCoeff;     // starvation coefficient
    float   beta;           // sigma update weight (centroid variance)
    float   lambda;         // pBelief weighting
    float   gamma;          // parent pBelief weighting
    float   temp;           // temperature for boltzmann normalization

    uint     winner;        // winning centroid index
    uint     genWinner;     // winning centroid index for generative procedure

    // node statistics
    float * mu;             // centroid locations ( a table nb x ns  )
    float * sigma;          // centroid variances
    float * starv;          // centroid starvation coefficients

    float   muSqDiff;

    long  * nCounts;        // number of observation counts
    
    // node input
    float * input;          // input pointer (null for input layer nodes)
    uint  * inputOffsets;   // offsets for each pixel taken from framePtr for this node
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
    uint layer;				// layer this node belongs to

    float * delta;           // vector that stores difference between observation and mu shared centroid vector
    /* HOST VARIABLES END */
} Node;

/* Node Functions Begin */
void  InitNode(                         // initialize a node.
                 uint,                  // node index
                 struct Destin *,              // reference to parent destin network
                 uint,                  // belief dimensionality (# centroids)
                 uint,                  // input dimensionality (# input values)
                 uint,                  // parent belief dimensionality
                 uint,                  // number of classes
                 uint,                  // ns = state dimensionality (number of inputs + number of beliefs) = ni +nb + np + nc
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
 
 
#endif

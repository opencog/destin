#ifndef LEARN_STRATS_H
#define LEARN_STRATS_H


#include "macros.h"

// Centroid learning strategy. Enumerates strategies for
// centroid movement / learning rate.
typedef  enum {
    CLS_FIXED,      // Centroid learn strategy fixed - fixed centroid learning rate
    CLS_DECAY,       // Centroid learn strategy decay learning rate decays with 1 / N where N = number
    CLS_DECAY_c1 // 2013.6.27, still under discussion!
                    // of times the centroid has won
} CentroidLearnStrat;

//function pointer to the centroid update / learning strategy
typedef float (*CentroidLearnStratFunc)(struct Destin *, struct Node *, uint, uint);


/** Sets the centroid learning strategy / rate
 * based on the CentroidLearnStrat enumeration
 */
void SetLearningStrat(
                 struct Destin *,           // destin structure to set learning strategy
                 CentroidLearnStrat         // learning strategy emum value
                 );


/** Centroid learning strategy Decay.
* Learning rate is 1/N where N is how many times the centroid has won.
* Corresponds to CLS_DECAY CentroidLearnStrat enum value.
* @returns the learning rate
*/
float CLS_Decay(
                struct Destin *,            // Destin network to calculate the learn rate for
                struct Node *,              // node to calculate learning rate for
                uint,                       // Which layer the node is in
                uint                        // Which centroid to calulate the learning rate for
                );

/** Centroid learning startegy Fixed.
 * Corresponds to CLS_DECAY CentroidLearnStrat enum value.
 * @returns constant ( currently hardcoded ) learning rate.
 */
float CLS_Fixed(
                struct Destin *,            // Destin network to calculate the learn rate for
                struct Node *,              // node to calculate learning rate for
                uint,                       // Which layer the node is in
                uint                        // Which centroid to calulate the learning rate for
                );

// 2013.6.27
// Still under discussion
float CLS_Decay_c1(
        struct Destin *,            // Destin network to calculate the learn rate for
        struct Node *,              // node to calculate learning rate for
        uint,                       // Which layer the node is in
        uint                        // Which centroid to calulate the learning rate for
        );

#endif

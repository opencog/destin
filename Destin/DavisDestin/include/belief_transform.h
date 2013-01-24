#ifndef DEST_BELIEF_TRANSFORM_H
#define DEST_BELIEF_TRANSFORM_H

struct Node;
struct Destin;

// centroid movement / learning rate.
typedef enum {
    DST_BT_BOLTZ,       // destin belief transform boltzmann
    DST_BT_P_NORM,   // destin belief transform exponential parameter
    DST_BT_NONE         // no transformation is applied
} BeliefTransformEnum;

/** Maps the string to the corresponding enum.
*   String values may be boltz, pnorm, or none.
*   If the string does not match one of those then DST_BT_NONE is returned.
*/
BeliefTransformEnum BeliefTransform_S_to_E(char * string);

//function pointer to the centroid update / learning strategy
typedef void (*BeliefTransformFunc)(struct Node *);


void SetBeliefTransform(struct Destin *, BeliefTransformEnum );



void DST_BT_Boltzmann(struct Node* n);
void DST_BT_PNorm(struct Node* n);
void DST_BT_None(struct Node* n);


#endif

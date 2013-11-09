#ifndef DEST_BELIEF_TRANSFORM_H
#define DEST_BELIEF_TRANSFORM_H

struct Node;
struct Destin;

/* This header defines the belief transform functions.
 * These belief transforms apply a transformation function
 * to the output belief vectors. For instance, applying the
 * power normalize transform with high temperature parameter can make
 * the belief distribution more spikey.
 */
/** Emumerates the possible belief transform functions.
  */
typedef enum {
    DST_BT_BOLTZ,       // destin belief transform boltzmann
    DST_BT_P_NORM,      // power normalization with an exponential parameter
    DST_BT_NONE,        // no transformation is applied
    DST_BT_WTA          // winner take all
} BeliefTransformEnum;

/** Maps the string to the corresponding enum.
*   String values may be boltz, pnorm, or none.
*   If the string does not match one of those then DST_BT_NONE is returned.
*/
BeliefTransformEnum BeliefTransform_S_to_E(char * string);

//function pointer to the belief transform functions
typedef void (*BeliefTransformFunc)(struct Node *);

/** Tells destin which function to be assigned to
  * the beliefTransform function pointer.
  * The belief transform function is applied to all
  * the nodes' beliefs after the beliefs are calulated.
  */
void SetBeliefTransform(struct Destin *, BeliefTransformEnum );

/** Boltzmann belief transform.
  * Corresponds to DST_BT_BOLTZ enum value.
  * Makes the nodes' belief distributions more spiked.
  * The higher the node's temperature (node->temp)
  * the more spiked it becomes.
  */
void DST_BT_Boltzmann(struct Node* n);

/** Power normalize.
  * Corresponds to the DST_BT_P_NORM enum value
  * Each element of the node's belief distribution
  * is raized to the P power where P is given by the
  * temperature of the node's layer ( given by the node->temp array).
  * Then the belief distribution is re-normalized to sum to 1.
  * If p < 1 then the distribution becomes more uniform.
  * If P=1 then it should remain unchanged. If P > 1 then the belief
  * distribution becomes more spike on the maximum elements.
  * It tries to make the nodes more decisive.
  */
void DST_BT_PNorm(struct Node* n);

/** No belief transform.
  * Corresponds to enum value DST_BT_NONE
  * Used when no post processing is to be done to the beliefs.
  */
void DST_BT_None(struct Node* n);

/** Winner take all.
  * The belief element with the maximum value gets 1.0,
  * while the others get 0.
  */
void DST_BT_WinnerTakeAll(struct Node* n);


#endif

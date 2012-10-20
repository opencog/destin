
#include "unit_test.h"
#include "destin.h"
#include <stdio.h>

#include <float.h>

int testInit(){

    uint ni, nl;
    ni = 2;
    nl = 1;
    uint nb [] = {1}; //1 centroid
    uint nc = 0;
    float beta = 1;
    float lambda = 1;
    float gamma = 1;
    float temperature [] = {1};
    float starvCoef = 0.1;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements);

    DestroyDestin(d);

    return 0;
}

int testFormulateNotCrash(){

    uint ni, nl;
    ni = 1;
    nl = 1;
    uint nb [] = {1}; //1 centroid
    uint nc = 0;
    float beta = 1;
    float lambda = 1;
    float gamma = 1;
    float temperature [] = {1};
    float starvCoef = 0.1;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements);

    float image [] = {0.0};
    FormulateBelief(d, image );

    DestroyDestin(d);

    return 0;
}

int testForumateStages(){
    uint ni, nl;
    ni = 1; //one dimensional centroid
    nl = 1;
    uint nb [] = {2}; //2 centroids
    uint nc = 0; // 0 classes
    float beta = 1;
    float lambda = 1;
    float gamma = 1;
    float temperature [] = {1};
    float starvCoef = 0.1;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements);

    float image [] = {0.55};
    int nid = 0; //node

    Node * n = &d->nodes[0];
    printf("ni: %i, nb: %i, np: %i, ns: %i, nc: %i\n", n->ni, n->nb, n->np, n->ns, n->nc);

    assertTrue(n->ni == ni);
    assertTrue(n->nb == nb[0]);
    assertTrue(n->np == 0); //no parents
    assertTrue(n->ns == (ni+nb[0]+0+nc));
    assertTrue(n->ns == 3);
    assertTrue(n->nc == 0); //# of classes

    assertTrue( n->inputOffsets != NULL);

    //get observation for the first node
    GetObservation( d->nodes, image, nid );
    //ni,previous belief * gamma,parent_prev_belief, class vector
    float exp1 [] = { /*input */ 0.55,
            0.5, 0.5 /*prev belief initialized to uniform  */
            /* parent belief, class vector*/};
    assertFloatArrayEquals(exp1, n->observation,3);

    CalculateDistances( d->nodes, nid );

    float exp_belief [] = {};

    //assertFloatArrayEquals(exp_belief,n->beliefEuc, nb[0] );


    NormalizeBeliefGetWinner( d->nodes, nid );
    UpdateWinner( d->nodes, d->inputLabel, nid );


    DestroyDestin(d);

    return 0;
}

int shouldFail(){
    assertTrue(1==2);
    return 0;
}

int main(int argc, char ** argv ){
    //RUN( shouldFail );

    RUN( testInit );
    RUN( testFormulateNotCrash );
    RUN( testForumateStages );
    printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");
}

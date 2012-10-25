
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
    float beta = 0.001;
    float lambda = 1;
    float gamma = 1;
    float temperature [] = {1};
    float starvCoef = 0.1;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements);

    float image [] = {0.55};
    int nid = 0; //node index

    Node * n = &d->nodes[0];
    printf("ni: %i, nb: %i, np: %i, ns: %i, nc: %i\n", n->ni, n->nb, n->np, n->ns, n->nc);

    assertTrue(n->ni == ni);
    assertTrue(n->ni == 1);
    assertTrue(n->nb == nb[0]);
    assertTrue(n->nb == 2);
    assertTrue(n->np == 0); //no parents
    assertTrue(n->ns == (ni+nb[0]+0+nc));
    assertTrue(n->ns == 3);
    assertTrue(n->nc == 0); //# of classes

    assertTrue( n->inputOffsets != NULL);

    //get observation for the first node
    GetObservation( d->nodes, image, nid );
    //ni,previous belief * gamma,parent_prev_belief, class vector
    float expected_obs [] = {   0.55,       /*input vector*/
                        0.5, 0.5    /*prev belief initialized to uniform  */
                        //none      /* parent belief vector*/
                        //none      /*class label vector*/
    };

    assertFloatArrayEquals( expected_obs , n->observation,3);


    memcpy( &n->mu[0 * n->ns], toFloatArray(3, 0.5, 0.5, 0.5), sizeof(float) * 3);
    memcpy( &n->mu[1 * n->ns], toFloatArray(3, 0.0, 0.5, 1.0), sizeof(float) * 3);

    //TODO: yea toFloatArray is a memory leak, but only a small one :D
    //will eventually make it clean itself up.
    assertFloatArrayEquals( toFloatArray(2, 1.0,1.0), n->starv,2 );

    assertFloatArrayEquals( toFloatArray(2, 0.5, 0.5), n->beliefEuc, 2 );
    CalculateDistances( d->nodes, nid );

    assertFloatArrayEqualsE( toFloatArray(2, 20.0, 1.345345588),n->beliefEuc, 2, .00001 );


    NormalizeBeliefGetWinner( d->nodes, nid );
    assertFloatArrayEqualsE( toFloatArray(2, 0.7055658715, 0.29443412),n->beliefEuc, 2, 0.000001 );
    assertFloatArrayEqualsE( toFloatArray(2, 0.7055658715, 0.29443412),n->pBelief, 2, 0.000001 );

    assertTrue(INIT_SIGMA == 0.00001);
    //TODO: combine toFloatArray and assertFloatArrayEquals
    assertFloatArrayEquals(toFloatArray(6, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA ), n->sigma, 6);
    UpdateWinner( d->nodes, d->inputLabel, nid );
    assertTrue( n->winner == 0 );
    assertFloatArrayEqualsE(toFloatArray(6, 0.00001249, 0.00000999, 0.00000999, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA ), n->sigma, 6, 1e-12);
    assertFloatArrayEqualsE(toFloatArray(2, 1.0, 0.9), n->starv, 2, 0.0);
    DestroyDestin(d);

    return 0;
}

int shouldFail(){
    assertTrue(1==2);
    return 0;
}


int testVarArgs(void)
{
   printFloatArray(toFloatArray(3, 9.0, 8.0, 7.0), 3);
   float *f = toFloatArray(2,1.2, 1.4);

   assertFloatEquals(1.2, f[0],1e-7);
   assertFloatEquals(1.4, f[1],1e-7);
   return 0;
}

int main(int argc, char ** argv ){

    //RUN( shouldFail );

    RUN(testVarArgs);
    RUN( testInit );
    RUN( testFormulateNotCrash );
    RUN( testForumateStages );
    printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");
}

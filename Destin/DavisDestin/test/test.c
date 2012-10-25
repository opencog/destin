
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


    assignFloatArray(&n->mu[0 * n->ns], 3, 0.5, 0.5, 0.5);
    assignFloatArray(&n->mu[1 * n->ns], 3, 0.0, 0.5, 1.0);

    //TODO: yea toFloatArray is a memory leak, but only a small one :D
    //will eventually make it clean itself up.
    assertFloatArrayEqualsEV(n->starv, 1e-12, 2, 1.0,1.0);

    assertFloatArrayEqualsEV(n->beliefEuc, 1e-12, 2, 0.5, 0.5);
    CalculateDistances( d->nodes, nid );

    assertFloatArrayEqualsEV( n->beliefEuc, 1e-5, 2, 20.0, 1.345345588);

    NormalizeBeliefGetWinner( d->nodes, nid );
    assertFloatArrayEqualsEV( n->beliefEuc, 1e-7, 2, 0.7055658715, 0.29443412);
    assertFloatArrayEqualsEV( n->pBelief, 1e-7, 2, 0.7055658715, 0.29443412);

    assertTrue(INIT_SIGMA == 0.00001);
    assertFloatArrayEqualsEV(n->sigma, 1e-12, 6, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);
    UpdateWinner( d->nodes, d->inputLabel, nid );
    assertTrue( n->winner == 0 );
    assertFloatArrayEqualsEV(n->sigma, 1e-12, 6, 0.00001249, 0.00000999, 0.00000999, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);
    //UpdateStarvation(d->nodes, nid);
    assertFloatArrayEqualsEV(n->starv, 0.0, 2, 1.0, 0.9);
    DestroyDestin(d);

    return 0;
}

int shouldFail(){
    assertTrue(1==2);
    return 0;
}



int testVarArgs(void){
    //test some of the unit test functions and macros
   printFloatArray(toFloatArray(3, 9.0, 8.0, 7.0), 3);
   float *f = toFloatArray(2,1.2, 1.4);

   assertFloatEquals(1.2, f[0],1e-7);
   assertFloatEquals(1.4, f[1],1e-7);

   assignFloatArray(f, 2, 0.2,0.3);
   assertFloatEquals(.2, f[0],1e-7);
   assertFloatEquals(.3, f[1],1e-7);

   assertFloatArrayEqualsEV(f, 1e-7, 2, 0.2, 0.3  );
   free(f);
   return 0;
}

int testUniform(){
    uint ni = 1; //input layer nodes cluster on 1 pixel input.
    uint nl = 2;
    uint nb [] = {4,4}; //4 shared centroids per layer
    uint nc = 0; // 0 classes
    float beta = 0.001;
    float lambda = 1;
    float gamma = 1;
    float temperature [] = {10};
    float starvCoef = 0.1;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements);
    assertTrue(MakeUniform(d)); //should repoint the node->mu pointers to the global mu list

    //should handle being made uniform twice
    assertTrue( MakeUniform(d));

    float image []  = {.11,.22,.33,.44};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];
    GetObservation( d->nodes, image, 0); //get observation for node 0 only
    assertFloatEquals( 0.11, n->observation[0], 1e-8);

    assignFloatArray(n->mu, 4, 0.5, 0.6, 0.7, 0.8);
    CalculateDistances(d->nodes, 0);

    NormalizeBeliefGetWinner( d->nodes, 0);//dont think need to change

    UpdateWinner( d->nodes, d->inputLabel, 0 );

    Uniform_UpdateStarvation(d->nodes, 0);


    DestroyDestin(d);
    return 0;
}

int main(int argc, char ** argv ){

    //RUN( shouldFail );

    RUN( testVarArgs );
    RUN( testInit );
    RUN( testFormulateNotCrash );
    RUN( testForumateStages );
    RUN( testUniform );
    printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");
}


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
    bool isUniform = false;
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
    printf("inited non uniform\n");
    DestroyDestin(d);
    printf("destroyed non uniform\n");
    
    //test uniform destin init
    isUniform = true;
    d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
    printf("Inited uniform.\n");
    DestroyDestin(d);
    printf("destroyed uniform.\n");

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
    bool isUniform = false;
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);

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
    bool isUniform = false;
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);

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


    assertFloatArrayEqualsEV(n->starv, 1e-12, 2, 1.0,1.0);//starv is initalized to 1.0

    assertFloatArrayEqualsEV(n->beliefEuc, 1e-12, 2, 0.5, 0.5);
    CalculateDistances( d->nodes, nid );

    assertFloatArrayEqualsEV( n->beliefEuc, 1e-5, 2, 20.0, 1.345345588);

    NormalizeBeliefGetWinner( d->nodes, nid );
    assertFloatArrayEqualsEV( n->beliefEuc, 1e-7, 2, 0.7055658715, 0.29443412);
    assertFloatArrayEqualsEV( n->pBelief, 1e-7, 2, 0.7055658715, 0.29443412);

    assertTrue(INIT_SIGMA == 0.00001);
    assertFloatArrayEqualsEV(n->sigma, 1e-12, 6, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);
    CalcCentroidMovement( d->nodes, d->inputLabel, nid );
    MoveCentroids(d->nodes, nid);
    assertTrue( n->winner == 0 );
    assertFloatArrayEqualsEV(n->sigma, 1e-12, 6, 0.00001249, 0.00000999, 0.00000999, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);
    UpdateStarvation(d->nodes, nid);
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
    
   //TODO: yea toFloatArray is a memory leak, but only a small one :D
   //will eventually make it clean itself up.
   printFloatArray(toFloatArray(3, 9.0, 8.0, 7.0), 3);
   float *f = toFloatArray(2,1.2, 1.4);

   assertFloatEquals(1.2, f[0],1e-7);
   assertFloatEquals(1.4, f[1],1e-7);

   assignFloatArray(f, 2, 0.2,0.3);
   assertFloatEquals(.2, f[0],1e-7);
   assertFloatEquals(.3, f[1],1e-7);

   assertFloatArrayEqualsEV(f, 1e-7, 2, 0.2, 0.3  );

   int an_int_array[] = {2, 4, 6, 8};
   assertIntArrayEqualsEV(an_int_array, 4, 2, 4, 6, 8);

   long a_long_array[] = {3, 6, 9, 12};
   assertLongArrayEqualsEV(a_long_array, 4, 3, 6, 9, 12);

   bool a_bool_array[] = {false, false, true, true};
   assertBoolArrayEqualsEV(a_bool_array, 4, false, false, true, true);


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
    float temperature [] = {10, 10};
    float starvCoef = 0.1;
    uint nMovements = 0;
    bool isUniform = true;
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
    assertTrue(d->isUniform);


    float image []  = {.11,.22,.88,.99};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];
    int nid;
    for(nid = 0; nid < 4 ; nid++){
        GetObservation( d->nodes, image, nid); //get observation for node 0 only
    }
    //spot check observations were made
    assertFloatEquals( 0.11, d->nodes[0].observation[0], 1e-8);
    assertFloatEquals( 0.99, d->nodes[3].observation[0], 1e-8);
    
    assertFloatArrayEqualsEV(n->observation, 1e-12, 1+4+4+0, 0.11, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
    assertFloatArrayEqualsEV(n[3].observation, 1e-12, 1+4+4+0, 0.99, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
   

    //set centroid locations
    //mu is a table nb x ns. ns = ni + nb + np + nc
    //nb = 4, ns = 9
    //all nodes point to the same centroids in a layer for uniform destin
    assignFloatArray(n->mu, 4 * 9, 
        0.05, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.06, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,  
        0.86, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 
        0.95, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);


    //make sure the mu pointers are shared between nodes
    assertTrue( d->nodes[0].mu == d->nodes[3].mu );

    //but not shared between layers
    assertFalse( d->nodes[0].mu == d->nodes[4].mu );

    //continue processing
    for(nid = 0; nid < 5 ; nid++){
        CalculateDistances(d->nodes, nid);
    }

    //manually calculate distance for node 0, centroid 0
    float c1 = 0.05;
    float c2 = 0.11;
    float dist = sqrt( (c2  - c1) * (c2 - c1) );
    assertFloatEquals( 1.0 / dist, d->nodes[0].beliefEuc[0], 9e-7);
    
    //manually calculate distance for node 0, centroid 3
    c1 = 0.95;
    dist = sqrt( (c2  - c1) * (c2 - c1) );
    assertFloatEquals( 1.0 / dist, d->nodes[0].beliefEuc[3], 6e-8);
    
    //manually calculate distance for node 3, centroid 0
    c1 = 0.05;
    c2 = 0.99;
    dist = sqrt( (c2  - c1) * (c2 - c1));
    assertFloatEquals( 1.0 / dist, d->nodes[3].beliefEuc[0], 2e-8);
    
    //manually calculate distance for node 3, centroid 3
    c1 = 0.95;
    c2 = 0.99;
    dist = sqrt( (c2  - c1) * (c2 - c1));
    assertFloatEquals( 1.0 / dist, d->nodes[3].beliefEuc[3], 6e-8);


    for(nid = 0 ; nid < 5 ; nid++){
        NormalizeBeliefGetWinner( d->nodes, nid);
    }
    
    //check that the right centroids won
    assertIntEquals(1, n[0].winner);
    assertIntEquals(1, n[1].winner);
    assertIntEquals(2, n[2].winner);
    assertIntEquals(3, n[3].winner);
    
    //CalcCentroidMovement( d->nodes, d->inputLabel, 0 );

    //Uniform_UpdateStarvation(d->nodes, 0);


    DestroyDestin(d);
    return 0;
}


int main(int argc, char ** argv ){

    //RUN( shouldFail );
    RUN(testVarArgs);
    RUN(testInit);
    RUN(testFormulateNotCrash);
    RUN(testForumateStages);
    RUN(testUniform);
    printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");
}

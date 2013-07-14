
#include <float.h>
#include <memory.h>
#include <stdio.h>

#include "destin.h"
#include "unit_test.h"
#include "cent_image_gen.h"

Destin * makeDestin(const int layers){
    if(layers > 8){
        printf("can't make more than 8 layers!\n");
    }
    uint ni = 16;
    uint nl = layers;


    uint nb [] = {2,2,2,2,2,2,2,2}; //centroids per layer
    uint nc = 0;
    float beta = 0.001;
    float lambda = 0.10;
    float gamma = 0.10;

    float temperature [] = {7.5, 8.5, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0};
    float starvCoef = 0.12;
    uint nMovements = 0;
    bool isUniform = true;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    return d;
}

int testInit(){

    uint ni, nl;
    ni = 16;
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
    float image[16] = {
        .01, .02, .03, .04,
        .05, .06, .07, .08,
        .09, .10, .11, .12,
        .13, .14, .15, .16
    };
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);

    Node * n = &d->nodes[0];

    assertTrue(n->ni == 16);
    assertFloatEquals(.01, image[n->inputOffsets[0]], 1e-8);
    assertFloatEquals(.02, image[n->inputOffsets[1]], 1e-8);
    assertFloatEquals(.16, image[n->inputOffsets[15]], 1e-8);


    printf("inited non uniform\n");
    DestroyDestin(d);

    printf("destroyed non uniform\n");
    
    //test uniform destin init
    isUniform = true;
    d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);

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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1;
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
    
   float * fa = toFloatArray(3, 9.0, 8.0, 7.0);
   printFloatArray(fa, 3);
   float *f = toFloatArray(2,1.2, 1.4);

   assertFloatEquals(1.2, f[0],1e-7);
   assertFloatEquals(1.4, f[1],1e-7);

   assignFloatArray(f, 2, 0.2,0.3);
   assertFloatEquals(.2, f[0],1e-7);
   assertFloatEquals(.3, f[1],1e-7);

   assertFloatArrayEqualsEV(f, 1e-7, 2, 0.2, 0.3  );

   int an_int_array[] = {2, 4, 6, 8};
   assertIntArrayEqualsV(an_int_array, 4, 2, 4, 6, 8);

   long a_long_array[] = {3, 6, 9, 12};
   assertLongArrayEqualsV(a_long_array, 4L, 3L, 6L, 9L, 12L);

   bool a_bool_array[] = {false, false, true, true};
   assertBoolArrayEqualsV(a_bool_array, 4, false, false, true, true);

   free(fa);
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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    assertTrue(d->isUniform);



    float image []  = {.11,.22,.88,.99};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];

    //set centroid locations
    //mu is a table nb x ns. ns = ni + nb + np + nc
    //nb = 4 (centroids), ns = 9
    //all nodes point to the same centroids in a layer for uniform destin
    assignFloatArray(n->mu, 4 * 9,
        0.05, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.06, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.86, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.95, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);

    int nid;
    for(nid = 0; nid < 5 ; nid++){
        GetObservation( d->nodes, image, nid); //get observation for node 0 only
    }
    //spot check observations were made
    assertFloatEquals( 0.11, d->nodes[0].observation[0], 1e-8);
    assertFloatEquals( 0.99, d->nodes[3].observation[0], 1e-8);
    
    assertFloatArrayEqualsEV(n[0].observation, 1e-12, 1+4+4+0, 0.11, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
    assertFloatArrayEqualsEV(n[1].observation, 1e-12, 1+4+4+0, 0.22, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
    assertFloatArrayEqualsEV(n[2].observation, 1e-12, 1+4+4+0, 0.88, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
    assertFloatArrayEqualsEV(n[3].observation, 1e-12, 1+4+4+0, 0.99, 0.25, 0.25, 0.25, 0.25, 0.25 , 0.25 , 0.25 , 0.25);
   

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

    Uniform_ResetStats(d);
    assertLongArrayEqualsV( d->uf_persistWinCounts[0], 4, 0L, 0L, 0L, 0L );
    for(nid = 0 ; nid < 5 ; nid++){
        NormalizeBeliefGetWinner( d->nodes, nid);
    }

    //centroid 0 wasn't chosen by any nodes, centroid 1 was chosen by 2 nodes but
    //the win count for a shared centriod only increments by 1 even if multiple nodes
    //pick it as winner
    assertLongArrayEqualsV( d->uf_persistWinCounts[0], 4, 0L, 1L, 1L, 1L );

    //check that the right centroids won
    assertIntEquals(1, n[0].winner);
    assertIntEquals(1, n[1].winner);
    assertIntEquals(2, n[2].winner);
    assertIntEquals(3, n[3].winner);
    
    for(nid = 0 ; nid < 5 ; nid++){
        CalcCentroidMovement( d->nodes, d->inputLabel, nid );
    }
    
    //check that deltas were created propertly
    //node 0 has winner 1
    //delta is observation vector - winning mu vector
    assertFloatArrayEqualsEV(n[0].delta, 1e-12, 9, 0.11 - 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assertFloatArrayEqualsEV(n[1].delta, 1e-12, 9, 0.22 - 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assertFloatArrayEqualsEV(n[2].delta, 2e-8,  9, 0.88 - 0.86, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assertFloatArrayEqualsEV(n[3].delta, 3e-8,  9, 0.99 - 0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    //TODO: test top node
    //TODO: test feedback over multiple iterations

    for(nid = 0 ; nid < 5 ; nid++){
        Uniform_AverageDeltas(d->nodes, nid);
    }

    
    assertIntArrayEqualsV(d->uf_winCounts[0], nb[0], 0, 2, 1, 1);
    assertFloatArrayEqualsEV(d->uf_avgDelta[0], 3e-8, nb[0] * d->nodes[0].ns,
        0.0,                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //average delta for shared centroid 0
        ((0.11 - 0.06) + (0.22 - 0.06)) / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ( 0.88 - 0.86 ),                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ( 0.99 - 0.95 ),                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);//average delta for shared centroid 3
    
    
    float layer0SharedSigma[nb[0] * d->nodes[0].ns];
    float layer1SharedSigma[nb[1] * d->nodes[1].ns];
    Uniform_ApplyDeltas(d, 0, layer0SharedSigma);
    Uniform_ApplyDeltas(d, 1, layer1SharedSigma);

    assertTrue(n[0].nCounts == NULL); //these are null in uniform destin

    //check that the shared centroids were moved to the correct positions
    assertFloatArrayEqualsEV(n[0].mu, 2e-8, 4 * 9,
        0.05,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 0, unchanged because it wasn't a winner
        0.165, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 1, averaged because two nodes picked it
                                                                   //averaged between node 0 and node 1 observations, .11 and .22 = .165
        0.88,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //moved directly to node 2 observation because only node 2 picked it
        0.99,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);//moved directly to node 3 observation because only node 3 picked it
    //Uniform_UpdateStarvation(d->nodes, 0);

    float * ad = d->uf_avgDelta[0];
    //calculate muSqDiff for layer 0
    float msq = ad[0*9] * ad[0*9] + ad[1*9] * ad[1*9] + ad[2*9] * ad[2*9] + ad[3*9] * ad[3*9];
    assertFloatEquals(msq, n[0].muSqDiff, 1e-12);

    DestroyDestin(d);
    return 0;
}

//same setup as testUniform, but call the main FormulateBelief function to make sure it calls everything in the correct order.
int testUniformFormulate(){

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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1; //turn on cluster training
    d->layerMask[1] = 1;

    float image []  = {.11,.22,.88,.99};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];

    //set centroid locations
    //mu is a table nb x ns. ns = ni + nb + np + nc
    //nb = 4 (centroids), ns = 9
    //all nodes point to the same centroids in a layer for uniform destin
    assignFloatArray(n->mu, 4 * 9,
        0.05, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.06, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.86, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.95, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);

    FormulateBelief(d, image);

    //check that the shared centroids were moved to the correct positions
    assertFloatArrayEqualsEV(n->mu, 2e-8, 4 * 9,
        0.05,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 0, unchanged because it wasn't a winner
        0.165, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 1, averaged because two nodes picked it
                                                                   //averaged between node 0 and node 1 observations, .11 and .22 = .165
        0.88,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //moved directly to node 2 observation because only node 2 picked it
        0.99,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);//moved directly to node 3 observation because only node 3 picked it

    assertFloatArrayEqualsEV(d->uf_starv[0], 1e-12, 4,
        1.0 - starvCoef, 1.0, 1.0, 1.0);

    assertFloatEquals(d->muSumSqDiff, n[0].muSqDiff + n[4].muSqDiff, 1e-12 );

    DestroyDestin(d);
    return 0;
}

int testSaveDestin1(){
    //test that SaveDestin and LoadDestin are working propertly

    uint ni = 16; //input layer nodes cluster on 4 pixel input.
    uint nl = 2;
    uint nb [] = {3,4}; //4 shared centroids per layer
    uint nc = 6; // 0 classes
    float beta = 0.001;
    float lambda = 0.96;
    float gamma = 0.78;
    float temperature [] = {7.5, 8.5};
    float starvCoef = 0.12;
    uint nMovements = 4;
    bool isUniform = true;
    uint ns0 = ni + nb[0] + nb[1] + nc;
    uint ns1 = 4*nb[0] + nb[1] + 0 + nc;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1;
    d->layerMask[1] = 1;

    SetLearningStrat(d, CLS_FIXED);
    assertTrue(ns0 == d->nodes[0].ns);
    assertTrue(ns1 == d->nodes[4].ns);

    uint maxNs = d->maxNs;
    uint nBeliefs = d->nBeliefs;

    //random image to apply to mix up the destin states to test serialization
    float image[] = {
        0.14, 0.23, 0.345, 0.432, 0.533, 0.6454, 0.7124, 0.8094, 0.14, 0.21, 0.345, 0.432, 0.534, 0.6454, 0.7124, 0.8094,
        0.15, 0.23, 0.345, 0.432, 0.532, 0.6454, 0.7127, 0.8094, 0.14, 0.22, 0.345, 0.432, 0.534, 0.6454, 0.7124, 0.8094,
        0.16, 0.23, 0.345, 0.432, 0.531, 0.6454, 0.7126, 0.8094, 0.14, 0.27, 0.345, 0.432, 0.534, 0.6454, 0.7124, 0.8094,
        0.17, 0.23, 0.345, 0.432, 0.530, 0.6454, 0.7248, 0.8094, 0.14, 0.24, 0.345, 0.432, 0.534, 0.6454, 0.7124, 0.8094
    };

    FormulateBelief(d, image);
    FormulateBelief(d, image);
    FormulateBelief(d, image);


    //backup uf_aveDelta
    int sizes[] =  {nb[0]  * ns0, nb[1] * ns1};
    float ** uf_avgDelta = copyFloatDim2Array(d->uf_avgDelta, 2, sizes);

    SaveDestin(d, "unit_test_destin.save");
    DestroyDestin(d);
    d = NULL;//must be set to NULL or LoadDestin will try to destroy an invalid pointer
    d = LoadDestin(d, "unit_test_destin.save");

    assertTrue(d->centLearnStrat == CLS_FIXED);
    assertTrue(d->nLayers == 2);
    assertIntArrayEqualsV(d->nb, 2, 3, 4);
    assertTrue(d->nc == 6);
    Node * n = &d->nodes[0];

    assertTrue(n->ni == 16);
    assertFloatEquals(0.001, n->beta, 1e-10);
    assertFloatEquals(0.96, n->nLambda, 1e-07); //accuracy is not very good
    assertFloatEquals( 0.78, n->gamma, 3e-8);
    assertFloatArrayEqualsEV(d->temp, 1e-12, 2, 7.5, 8.5 );
    assertTrue(n->starvCoeff == starvCoef);
    assertTrue(d->nMovements == nMovements);
    assertTrue(d->isUniform == isUniform);

    assertIntArrayEqualsV(d->layerMask, 2, 0, 0); //TODO: layer mask is not saved,should be 1, 1
    assertIntArrayEqualsV(d->layerSize, 2, 4, 1);
    assertTrue(d->maxNb == 4);
    assertTrue(d->maxNs == maxNs);
    assertFloatEquals(0.0, d->muSumSqDiff, 0); //it currently resets to 0
    assertTrue(d->nBeliefs == nBeliefs);

    assertFloatArrayEqualsE(uf_avgDelta[0], d->uf_avgDelta[0], nb[0] * ns0, 0.0  );
    assertFloatArrayEqualsE(uf_avgDelta[1], d->uf_avgDelta[1], nb[1] * ns1, 0.0  );

    return 0;
}

void turnOnMask(Destin * d){
    int i;
    for(i = 0 ; i < d->nLayers ;i++){
        d->layerMask[i] = 1;
    }
}

float ** makeRandomImages(uint image_size, uint nImages){

    //generate random images
    uint i, j;
    float ** images;
    MALLOC(images, float *, nImages);

    for(i = 0 ; i < nImages; i++){
        MALLOC(images[i], float, image_size);
    }

    for(i = 0 ; i < image_size ; i++){
        for(j = 0 ; j < nImages ; j++){
            images[j][i] = (float)rand()  / (float)RAND_MAX;
        }
    }
    return images;
}

void freeRandomImages(float ** images, uint nImages){
    uint i;
    for(i = 0 ; i < nImages ; i++){
        FREE(images[i]);
    }
    FREE(images);
    return;
}

int _testSaveDestin2(bool isUniform, CentroidLearnStrat learningStrat, BeliefTransformEnum bt){
    //Test that SaveDestin and LoadDestin are working propertly.
    //This uses the strategy of checking that the belief outputs are the same
    //after loading a saved destin and repeating the same input image.

    uint ni = 16; //input layer nodes cluster on 4 pixel input.
    uint nl = 4;
    uint nb [] = {3, 4, 2, 4}; //4 shared centroids per layer
    uint nc = 6; // 0 classes
    float beta = 0.001;
    float lambda = 0.56;
    float gamma = 0.28;
    float temperature [] = {3.5, 4.5, 5.0, 4.4};
    float starvCoef = 0.12;
    uint nMovements = 4;
    uint i, j;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, bt);
    turnOnMask(d);

    //generate random images
    uint image_size = ni * d->layerSize[0];
    uint nImages = 5;
    float ** images = makeRandomImages(image_size, nImages);

    //mix up destin
    for(i = 0 ; i < 5; i++){
        for(j = 0 ; j < nImages ; j++){
            FormulateBelief(d, images[j]);
        }
    }

    //save it
    SaveDestin(d, "testSaveDestin2.save");

    //mix it up some more
    uint iterations = 50;
    for(i = 0 ; i < iterations; i++){
        for(j = 0 ; j < nImages ; j++){
            assertNoNans(d->belief, d->nBeliefs); //test that non nans (float "not a number") are occuring
            FormulateBelief(d, images[j]);
        }
    }

    //back up its output state so it can be compared later
    float * beliefState1;
    uint nBeliefs = d->nBeliefs;
    MALLOC(beliefState1, float, d->nBeliefs);
    memcpy(beliefState1, d->belief, sizeof(float) * d->nBeliefs);

    DestroyDestin(d);
    d = NULL;

    //restore destin then reapply the same observations, should end up with the same
    //belief state at the end if it was restored properly
    d = LoadDestin(d, "testSaveDestin2.save");
    turnOnMask(d);

    //reapply same observations
    for(i = 0 ; i < iterations; i++){
        for(j = 0 ; j < nImages ; j++){
            FormulateBelief(d, images[j]);
        }
    }

    assertTrue(d->nBeliefs == nBeliefs);
    //check that the same observations lead to the same belief outputs
    assertFloatArrayEquals(beliefState1, d->belief, nBeliefs);

    DestroyDestin(d);

    FREE(beliefState1);
    freeRandomImages(images, nImages);
    return 0;
}

int testSaveDestin2(){
    assertTrue(_testSaveDestin2(true, CLS_FIXED, DST_BT_BOLTZ) == 0); //uniform on
    assertTrue(_testSaveDestin2(false, CLS_FIXED, DST_BT_BOLTZ) == 0);//uniform off
    assertTrue(_testSaveDestin2(false, CLS_FIXED, DST_BT_P_NORM) == 0);//uniform off
    assertTrue(_testSaveDestin2(true, CLS_DECAY, DST_BT_P_NORM) == 0);
    assertTrue(_testSaveDestin2(false, CLS_DECAY, DST_BT_BOLTZ) == 0);
    return 0;
}

int testLoadFromConfig(){
    //TODO: add configuration for the learning strategy
    Destin * d = CreateDestin("testconfig.conf");
    assertTrue(d->isUniform == true);
    assertIntArrayEqualsV(d->nb, 3, 4, 5, 6);
    assertFloatArrayEqualsEV(d->temp, 1e-12, 3, 3.1, 3.2, 3.3 );
    assertTrue(d->nc == 8);
    assertTrue(d->nMovements == 7);
    assertTrue(d->nLayers == 3);
    Node * n = &d->nodes[0];
    assertTrue(n->ni == 16);
    assertFloatEquals(0.002, n->beta, 1e-8);
    assertFloatEquals(0.1, n->nLambda, 1e-8);
    assertFloatEquals(0.2, n->gamma, 1e-8);
    assertFloatEquals(0.001, n->starvCoeff, 1e-8);
    assertTrue(d->beliefTransform == DST_BT_BOLTZ);
    DestroyDestin(d);
    return 0;
}


//test GenerateInputFromBelief to make sure it doesn't crash
int _testGenerateInputFromBelief(bool isUniform){

    uint ni = 16; //input layer nodes cluster on 4 pixel input.
    uint nl = 4;
    uint nb [] = {3,4,2,2}; //centroids per layer
    uint nc = 0;
    float beta = 0.001;
    float lambda = 0.10;
    float gamma = 0.10;
    float temperature [] = {7.5, 8.5, 4.0, 4.0};
    float starvCoef = 0.12;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    d->layerMask[0] = 1;
    d->layerMask[1] = 1;
    d->layerMask[2] = 1;
    d->layerMask[3] = 1;

    float *outFrame;
    MALLOC( outFrame, float, d->layerSize[0] * d->nodes[0].ni);
    uint i, nImages = 4;
    float ** images = makeRandomImages(d->layerSize[0] * d->nodes[0].ni, nImages);

    for(i = 0 ; i < 8; i++){
        FormulateBelief(d, images[i % nImages]);
    }

    GenerateInputFromBelief(d, outFrame);

    FREE(outFrame);
    DestroyDestin(d);
    freeRandomImages(images, nImages);
    return 0;
}


int testGenerateInputFromBelief(){
    assertTrue(_testGenerateInputFromBelief(false) == 0 );
    assertTrue(_testGenerateInputFromBelief(true) == 0 );
    return 0;
}

int testGetNode(){
    uint ni = 1; //input layer nodes cluster on 4 pixel input.
    uint nl = 4;
    uint nb [] = {2,2,2,2}; //centroids per layer
    uint nc = 0;
    float beta = 0.001;
    float lambda = 0.10;
    float gamma = 0.10;
    float temperature [] = {7.5, 8.5, 4.0,3.3};
    float starvCoef = 0.12;
    uint nMovements = 0;

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, false, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);

    //set node outputs
    int i;
    for(i = 0; i < d->nInputPipeline; i++){
        d->belief[i] = i;
    }

    // copy node's belief to parent node's input
    memcpy( d->inputPipeline, d->belief, sizeof(float)*d->nInputPipeline );
    float image[] = {
        0.1, 0.2, 0.3, 0.4,
        0.11, 0.21, 0.31, 0.41,
        0.12, 0.22, 0.32, 0.42,
        0.13, 0.23, 0.33, 0.43,
    };
    //FormulateBelief(d, image );
    //FormulateBelief(d, image );
    //defines the expected mapping between parents and children
    int matches [] = {
      //pin   cout cr, cc, pr,pc,player
        0,    0,  0,  0,  0,  0, 1,
        1,    1,  0,  0,  0,  0, 1,
        2,    0,  0,  1,  0,  0, 1,
        3,    1,  0,  1,  0,  0, 1,
        4,    0,  1,  0,  0,  0, 1,
        5,    1,  1,  0,  0,  0, 1,
        6,    0,  1,  1,  0,  0, 1,
        7,    1,  1,  1,  0,  0, 1,
//2nd parent node
        0,    0,  0,  2,  0,  1, 1,
        1,    1,  0,  2,  0,  1, 1,
        2,    0,  0,  3,  0,  1, 1,
        3,    1,  0,  3,  0,  1, 1,
        4,    0,  1,  2,  0,  1, 1,
        5,    1,  1,  2,  0,  1, 1,
        6,    0,  1,  3,  0,  1, 1,
        7,    1,  1,  3,  0,  1, 1,
//3nd parent node
        0,    0,  2,  0,  1,  0, 1,
        1,    1,  2,  0,  1,  0, 1,
        2,    0,  2,  1,  1,  0, 1,
        3,    1,  2,  1,  1,  0, 1,
        4,    0,  3,  0,  1,  0, 1,
        5,    1,  3,  0,  1,  0, 1,
        6,    0,  3,  1,  1,  0, 1,
        7,    1,  3,  1,  1,  0, 1,
//4nd parent node
        0,    0,  2,  2,  1,  1, 1,
        1,    1,  2,  2,  1,  1, 1,
        2,    0,  2,  3,  1,  1, 1,
        3,    1,  2,  3,  1,  1, 1,
        4,    0,  3,  2,  1,  1, 1,
        5,    1,  3,  2,  1,  1, 1,
        6,    0,  3,  3,  1,  1, 1,
        7,    1,  3,  3,  1,  1, 1,
//top layer parent node
        0,    0,  0,  0,  0,  0, 2,
        1,    1,  0,  0,  0,  0, 2,
        2,    0,  0,  1,  0,  0, 2,
        3,    1,  0,  1,  0,  0, 2,
        4,    0,  1,  0,  0,  0, 2,
        5,    1,  1,  0,  0,  0, 2,
        6,    0,  1,  1,  0,  0, 2,
        7,    1,  1,  1,  0,  0, 2,
    };
    int m;
    for(m = 0; m < 40 ; m++){
        printf("m: %i\n", m);
        int rs = 7;
        int pi = matches[m*rs+0];//parent input element
        int co = matches[m*rs+1];//child output belief element
        int cr = matches[m*rs+2];//child row
        int cc = matches[m*rs+3];//child col
        int pr = matches[m*rs+4];//parent row
        int pc = matches[m*rs+5];//parent layer
        int pl = matches[m*rs+6];//parent layer
        Node * parent_node = GetNodeFromDestin(d, pl, pr, pc);
        assertFloatEquals(parent_node->input[pi], GetNodeFromDestin(d, pl - 1,cr, cc)->pBelief[co], 1e-12);
    }


    float * obs = d->nodes[0].observation;
    int l;
    //try to find if there are overlapping pointers because it looks li
    for(l = 0; l < nl ; l++){
        //for(n = 0; n < d->layerSize[l]; n++){

        //}
    }

    DestroyDestin(d);
    return 0;
}



int test8Layers(){
    Destin * d = makeDestin(8);

    assertIntEquals(43690, d->nBeliefs);
    assertIntEquals(43688, d->nInputPipeline);
    assertIntEquals(16384, d->layerSize[0]);

    assertIntEquals(0, d->layerNodeOffsets[0]);
    assertIntEquals(16384, d->layerNodeOffsets[1]);
    assertIntEquals(20480, d->layerNodeOffsets[2]);
    assertIntEquals(21504, d->layerNodeOffsets[3]);
    assertIntEquals(21844, d->layerNodeOffsets[7]);

    assertIntEquals(16384, GetNodeFromDestin(d, 1, 0, 0)->nIdx);
    assertIntEquals(20480, GetNodeFromDestin(d, 2, 0, 0)->nIdx);
    assertIntEquals(21504, GetNodeFromDestin(d, 3, 0, 0)->nIdx);
    assertIntEquals(21844, GetNodeFromDestin(d, 7, 0, 0)->nIdx);

    DestroyDestin(d);
    return 0;
}


int testInputOffsets(){
    //check some parent nodes that they have the right input offsets to their child nodes' output belief vectors

    Destin * d = makeDestin(4);
    assertIntEquals(2, GetNodeFromDestin(d, 0, 0, 2)->nIdx);
    assertIntEquals(44, GetNodeFromDestin(d, 0, 5, 4)->nIdx);
    assertIntEquals(64, GetNodeFromDestin(d, 1, 0, 0)->nIdx);
    assertIntEquals(80, GetNodeFromDestin(d, 2, 0, 0)->nIdx);

    //spot check some parent nodes in layer 1 that they have the right input offsets to their child nodes' output belief vectors
    assertIntArrayEqualsV(GetNodeFromDestin(d, 1, 0, 0)->inputOffsets, 8, 0, 1, 2, 3, 16, 17, 18, 19);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 1, 0, 2)->inputOffsets, 8, 8, 9, 10, 11, 24, 25, 26, 27);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 1, 3, 1)->inputOffsets, 8, 100, 101, 102, 103, 116, 117, 118, 119);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 1, 3, 2)->inputOffsets, 8, 104, 105, 106, 107, 120, 121, 122, 123);

    //same but for layer 2
    assertIntArrayEqualsV(GetNodeFromDestin(d, 2, 0, 0)->inputOffsets, 8, 128, 129, 130, 131, 136, 137, 138, 139);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 2, 0, 1)->inputOffsets, 8, 132, 133, 134, 135, 140, 141, 142, 143);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 2, 1, 0)->inputOffsets, 8, 144, 145, 146, 147, 152, 153, 154, 155);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 2, 1, 1)->inputOffsets, 8, 148, 149, 150, 151, 156, 157, 158, 159);

    //same but for the top layerr
    assertIntArrayEqualsV(GetNodeFromDestin(d, 3, 0, 0)->inputOffsets, 8, 160, 161, 162, 163, 164, 165, 166, 167);

    DestroyDestin(d);

    return 0;
}


int  testLinkParentBeliefToChildren(){
    // Test that parent nodes have the right children in their children pointers
    Destin * d = makeDestin(4);

    Node * parent = GetNodeFromDestin(d, 1, 0, 2);

    assertIntEquals(66, parent->nIdx);
    assertIntEquals(4, parent->children[0]->nIdx);
    assertIntEquals(5, parent->children[1]->nIdx);
    assertIntEquals(12, parent->children[2]->nIdx);
    assertIntEquals(13, parent->children[3]->nIdx);

    //check some child nodes point to the correct parent
    assertTrue(GetNodeFromDestin(d, 0, 4, 7)->parent_pBelief == GetNodeFromDestin(d, 1, 2, 3)->pBelief);
    assertTrue(GetNodeFromDestin(d, 0, 6, 4)->parent_pBelief == GetNodeFromDestin(d, 1, 3, 2)->pBelief);
    assertTrue(GetNodeFromDestin(d, 0, 4, 3)->parent_pBelief == GetNodeFromDestin(d, 1, 2, 1)->pBelief);

    parent = GetNodeFromDestin(d, 1, 2, 2);
    assertTrue(parent->children[3]->parent_pBelief == parent->pBelief);



    DestroyDestin(d);
    return 0;
}

int testCentroidImageGeneration(){
    uint ni = 1; //input layer nodes cluster on 1 pixel input.
    uint nl = 3;
    uint nb [] = {2,2,2}; //centroids per layer
    uint nc = 0;
    float beta = 0.001;
    float lambda = 0.10;
    float gamma = 0.10;
    float temperature [] = {7.5, 8.5, 4.0,3.3};
    float starvCoef = 0.12;
    uint nMovements = 0;
    bool isUniform = true;
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform, 1);
    SetBeliefTransform(d, DST_BT_BOLTZ);

    Node * n = GetNodeFromDestin(d, 0, 0 ,0);
    n->mu[0 * n->ns + 0] = 0.0;
    n->mu[1 * n->ns + 0] = 1.0; // black ( or is it white?)

    n = GetNodeFromDestin(d, 1, 0 ,0);
    assignFloatArray(&n->mu[0], 8,          0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); //all black
    assignFloatArray(&n->mu[1 * n->ns], 8,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0); //all white

    n = GetNodeFromDestin(d, 2, 0 ,0);

    //top half black, bottom half white
    assignFloatArray(&n->mu[0], 8,          1.0, 0.0, 1.0, 0.0,
                                            0.0, 1.0, 0.0, 1.0);

    //all black, bottom right dark grey
    assignFloatArray(&n->mu[1 * n->ns], 8,  1.0, 0.0, 1.0, 0.0,
                                            1.0, 0.0, 0.75, 0.25);


    float *** images = Cig_CreateCentroidImages(d, 1.0);

    // check that the generated images are correct
    assertFloatArrayEqualsEV(images[0][0], 0.0, 1, 0.0);
    assertFloatArrayEqualsEV(images[0][1], 0.0, 1, 1.0);

    assertFloatArrayEqualsEV(images[1][0], 0.0, 4, 1.0, 1.0, 1.0, 1.0);
    assertFloatArrayEqualsEV(images[1][1], 0.0, 4, 0.0, 0.0, 0.0, 0.0);

    assertFloatArrayEqualsEV(images[2][0], 0.0, 16,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0,
                             0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0);

    assertFloatArrayEqualsEV(images[2][1], 0.0, 16,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 0.75, 0.75,
                             1.0, 1.0, 0.75, 0.75);

    float aDist[3] = {1.0,2.0,3.0};
    float aDistNormed[3];
    Cig_PowerNormalize(aDist, aDistNormed, 3, 2);
    assertFloatArrayEqualsEV(aDistNormed, 1e-6, 3, 0.0714285714, 0.2857142857, 0.6428571429 );

    //try if the source is same as dest
    Cig_PowerNormalize(aDist, aDist, 3, 2);
    assertFloatArrayEqualsEV(aDist, 1e-6, 3, 0.0714285714, 0.2857142857, 0.6428571429 );

    Cig_DestroyCentroidImages(d, images);
    DestroyDestin(d);

    return 0;
}

int main(int argc, char ** argv ){

    //RUN( shouldFail );

    RUN(testVarArgs);

    RUN(testInit);
    RUN(testInputOffsets);
    RUN(testFormulateNotCrash);
    RUN(testForumateStages);
    RUN(testUniform);
    RUN(testUniformFormulate);
    RUN(testSaveDestin1);
    RUN(testSaveDestin2);
    RUN(testLoadFromConfig);
    RUN(testGenerateInputFromBelief);

    RUN(test8Layers);

    RUN(testLinkParentBeliefToChildren);

    //RUN(testGetNode); //TODO: fix and renable this test
    RUN(testCentroidImageGeneration);

    UT_REPORT_RESULTS();

    return 0;
}


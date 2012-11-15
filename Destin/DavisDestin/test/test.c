
#include "unit_test.h"
#include "destin.h"
#include <stdio.h>

#include <float.h>
#include <memory.h>

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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);

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
   assertIntArrayEqualsV(an_int_array, 4, 2, 4, 6, 8);

   long a_long_array[] = {3, 6, 9, 12};
   assertLongArrayEqualsV(a_long_array, 4, 3, 6, 9, 12);

   bool a_bool_array[] = {false, false, true, true};
   assertBoolArrayEqualsV(a_bool_array, 4, false, false, true, true);


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
    //TODO: rename unit test array equals macros

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
    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
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

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
    d->layerMask[0] = 1;
    d->layerMask[1] = 1;

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

    assertTrue(d->nLayers == 2);
    assertIntArrayEqualsV(d->nb, 2, 3, 4);
    assertTrue(d->nc == 6);
    Node * n = &d->nodes[0];

    assertTrue(n->ni == 16);
    assertFloatEquals(0.001, n->beta, 1e-10);
    assertFloatEquals(0.96, n->lambda, 1e-07); //accuracy is not very good
    assertFloatEquals( 0.78, n->gamma, 3e-8);
    assertFloatArrayEqualsEV(d->temp, 1e-12, 2, 7.5, 8.5 );
    assertTrue(n->starvCoeff == starvCoef);
    assertTrue(d->nMovements == nMovements);
    assertTrue(d->isUniform == isUniform);

    assertIntArrayEqualsV(d->layerMask, 2, 0, 0); //TODO: layer mask is not saved,should be 1, 1
    assertIntArrayEqualsV(d->layerSize, 2, 4, 1);
    assertTrue(d->maxNb == 4);
    assertTrue(d->maxNs == maxNs);
    assertTrue(d->muSumSqDiff == 0.0);
    assertTrue(d->nBeliefs == nBeliefs);

    assertFloatArrayEqualsE(uf_avgDelta[0], d->uf_avgDelta[0], nb[0] * ns0, 0.0  );
    assertFloatArrayEqualsE(uf_avgDelta[1], d->uf_avgDelta[1], nb[1] * ns1, 0.0  );
    //printFloatArray(d->uf_avgDelta[0], nb[0] * ns0 );
    //printFloatArray(d->uf_avgDelta[1], nb[1] * ns1 );
    //TODO: finish unit testing load and save

    return 0;
}

void turnOnMask(Destin * d){
    int i;
    for(i = 0 ; i < d->nLayers ;i++){
        d->layerMask[i] = 1;
    }
}

int _testSaveDestin2(bool isUniform){
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

    Destin * d = InitDestin(ni, nl, nb, nc, beta, lambda, gamma, temperature, starvCoef, nMovements, isUniform);
    turnOnMask(d);

    //generate random images
    uint image_size = ni * d->layerSize[0];
    uint nImages = 5;
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
            assertNoNans(d->belief, d->nBeliefs); //test that non nans are occuring
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
    for(j = 0 ; j < nImages ; j++){
        FREE(images[j]);
    }
    FREE(images);
    FREE(beliefState1);
    return 0;
}

int testSaveDestin2(){
    assertTrue(_testSaveDestin2(true) == 0); //is uniform on
    assertTrue(_testSaveDestin2(false) == 0);//is uniform off
}


int main(int argc, char ** argv ){

    //RUN( shouldFail );
    RUN(testVarArgs);
    RUN(testInit);
    RUN(testFormulateNotCrash);
    RUN(testForumateStages);
    RUN(testUniform);
    RUN(testUniformFormulate);
    RUN(testSaveDestin1);
    RUN(testSaveDestin2);
    printf("FINSHED TESTING: %s\n", TEST_HAS_FAILURES ? "FAIL" : "PASS");
}

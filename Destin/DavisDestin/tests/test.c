#include <float.h>
#include <memory.h>
#include <stdio.h>

#include "destin.h"
#include "unit_test.h"
#include "cent_image_gen.h"
#include "array.h"

/*
 * This file contains many unit tests.
 * Each unit test is run with the RUN() macro in the main() function.
 * See unit_test.h to see what the RUN() function does.
 * If a unit test passes, its function returns 0.
 * The unit tests consists of assert statements.
 * If an assert statement passes, the unit test continues. If the assert
 * statement fails then it forces an exit of the function by returning 1.
 * The UT_REPORT_RESULTS() macro at the end of the main() function
 * will report if there were any unit tests failures. If there were
 * any failures the unit test executable exists with 1, which can
 * be detected in a shell script as a failure. See Destin/run_tests.sh
 */

Destin * makeDestinFromLayerCfg(uint layers, uint inputDim, uint * layerWidths)
{
    DestinConfig * dc = CreateDefaultConfig(layers);
    dc->inputDim = inputDim;
    if(layerWidths != NULL){
        memcpy(dc->layerWidths, layerWidths, sizeof(uint) * layers);
    }

    Destin * d = InitDestinWithConfig(dc);
    SetBeliefTransform(d, DST_BT_BOLTZ);
    DestroyConfig(dc);
    return d;
}

Destin * makeDestin(uint layers){
    return makeDestinFromLayerCfg(layers, 16, NULL);
}

int testInit(){

    uint nl = 1;

    float image[16] = {
        .01, .02, .03, .04,
        .05, .06, .07, .08,
        .09, .10, .11, .12,
        .13, .14, .15, .16
    };

    DestinConfig * dc = CreateDefaultConfig(nl);
    dc->centroids[0] = 1;
    dc->isUniform = false;

    Destin * d = InitDestinWithConfig(dc);
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
    dc->isUniform = true;
    d = InitDestinWithConfig(dc);
    SetBeliefTransform(d, DST_BT_BOLTZ);

    printf("Inited uniform.\n");
    DestroyDestin(d);
    printf("destroyed uniform.\n");

    DestroyConfig(dc);
    return 0;
}

int testFormulateNotCrash(){
    uint nl;
    nl = 1;

    DestinConfig * dc = CreateDefaultConfig(nl);
    dc->centroids[0] = 1;
    dc->isUniform = false;
    dc->inputDim = 1;

    Destin * d = InitDestinWithConfig(dc);

    SetBeliefTransform(d, DST_BT_BOLTZ);
    float image [] = {0.0};
    FormulateBelief(d, image );

    DestroyConfig(dc);
    DestroyDestin(d);

    return 0;
}

int testFormulateStages(){
    DestinConfig * dc = CreateDefaultConfig(1);
    dc->inputDim = 1; //one dimensional centroid
    dc->centroids[0] = 2; //2 centroids
    dc->beta = 0.001;
    dc->lambdaCoeff = 1.0;
    dc->gamma = 1.0;
    dc->temperatures[0] = 5.0;
    dc->starvCoeff = 0.1;
    dc->freqCoeff = 0.05;
    dc->freqTreshold = 0;
    dc->isUniform = false;

    Destin * d = InitDestinWithConfig(dc);


    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1;
    float image [] = {0.55};
    int nid = 0; //node index

    Node * n = &d->nodes[0];
    printf("ni: %i, nb: %i, np: %i, ns: %i, nc: %i\n", n->ni, n->nb, n->np, n->ns, n->nc);

    assertTrue(n->ni == dc->inputDim);
    assertTrue(n->ni == 1);
    assertTrue(n->nb == dc->centroids[0]);
    assertTrue(n->nb == 2);
    assertTrue(n->np == 0); //no parents
    assertTrue(n->ns == (dc->inputDim + dc->centroids[0] +0+dc->nClasses));
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


    assignFloatArray(n->mu[0], 3, 0.5, 0.5, 0.5);
    assignFloatArray(n->mu[1], 3, 0.0, 0.5, 1.0);


    assertFloatArrayEqualsEV(n->starv, 1e-12, 2, 1.0,1.0);//starv is initalized to 1.0

    assertTrue(INIT_SIGMA == 0.00001);
    assertFloatArrayEqualsE2DV(n->sigma, 1e-5, 2, 3, INIT_SIGMA, INIT_SIGMA, INIT_SIGMA,
                                                     INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);

    assertFloatArrayEqualsEV(n->beliefEuc, 1e-12, 2, 0.5, 0.5);
    assertFloatArrayEqualsEV(n->beliefMal, 1e-12, 2, 0.5, 0.5);
    CalculateDistances( d->nodes, nid );

    assertFloatArrayEqualsEV(n->beliefEuc, 1e-5, 2, 0.9523809524, 0.5736236037);
    assertFloatArrayEqualsEV(n->beliefMal, 1e-5, 2, 0.0594834872, 0.0042363334);

    NormalizeBeliefGetWinner( d->nodes, nid );
    assertFloatArrayEqualsEV( n->beliefEuc, 1e-7, 2, 0.775739751, 0.224260249);
    assertFloatArrayEqualsEV( n->beliefMal, 1e-7, 2, 0.9870696376, 0.0129303624);

#ifdef USE_EUC
    assertFloatArrayEqualsEV( n->beliefEuc, 1e-7, 2, 0.775739751, 0.224260249);
#endif
#ifdef USE_MAL
    assertFloatArrayEqualsEV( n->beliefMal, 1e-7, 2, 0.9870696376, 0.0129303624);
#endif

    CalcCentroidMovement( d->nodes, d->inputLabel, nid );
    MoveCentroids(d->nodes, nid);
    assertTrue( n->winner == 0 );
    assertFloatArrayEqualsE2DV(n->sigma, 1e-5, 2, 3, 0.00001249, 0.00000999, 0.00000999,
                                                     INIT_SIGMA, INIT_SIGMA, INIT_SIGMA);
    UpdateStarvation(d->nodes, nid);
    assertFloatArrayEqualsEV(n->starv, 0.0, 2, 1.0, 0.9);
    DestroyDestin(d);
    DestroyConfig(dc); dc = NULL;

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
   print_float_array(fa, 3);

   float *f = toFloatArray(2,1.2, 1.4);

   assertFloatEquals(1.2, f[0],1e-7);
   assertFloatEquals(1.4, f[1],1e-7);

   assignFloatArray(f, 2, 0.2,0.3);
   assertFloatEquals(.2, f[0],1e-7);
   assertFloatEquals(.3, f[1],1e-7);

   assertFloatArrayEqualsEV(f, 1e-7, 2, 0.2, 0.3  );

   float * f2d[3];
   f2d[0] = toFloatArray(4, 0.1, 0.2, 0.3, 0.4);
   f2d[1] = toFloatArray(4, 1.2, 1.3, 1.4, 1.5);
   f2d[2] = toFloatArray(4, 2.3, 2.4, 2.5, 2.6);
   assertFloatArrayEqualsE2DV(f2d, 1e-7, 3, 4, 0.1, 0.2, 0.3, 0.4,
                                               1.2, 1.3, 1.4, 1.5,
                                               2.3, 2.4, 2.5, 2.6);

   int an_int_array[] = {2, 4, 6, 8};
   assertIntArrayEqualsV(an_int_array, 4, 2, 4, 6, 8);

   long a_long_array[] = {3, 6, 9, 12};
   assertLongArrayEqualsV(a_long_array, 4L, 3L, 6L, 9L, 12L);

   print_long_array(a_long_array, 4);

   bool a_bool_array[] = {false, false, true, true};
   assertBoolArrayEqualsV(a_bool_array, 4, false, false, true, true);

   free(f2d[0]);
   free(f2d[1]);
   free(f2d[2]);
   free(fa);
   free(f);
   return 0;
}


int testUniform(){
    DestinConfig * dc = CreateDefaultConfig(2);

    assignUIntArray(dc->centroids, 2, 4, 4);
    dc->inputDim = 1;
    dc->lambdaCoeff = 1;
    dc->gamma = 1;
    assignFloatArray(dc->temperatures, 2, 10.0, 10.0);
    dc->starvCoeff = 0.1;
    dc->isUniform = true;

    Destin * d = InitDestinWithConfig(dc);

    SetBeliefTransform(d, DST_BT_BOLTZ);
    assertTrue(d->isUniform);

    float image []  = {.11,.22,.88,.99};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];

    //set centroid locations
    //mu is a table nb x ns. ns = ni + nb + np + nc
    //nb = 4 (centroids), ns = 9
    //all nodes point to the same centroids in a layer for uniform destin
    assignFloatArray2D(n->mu, 4, 9,
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
    assertFloatEquals( 1.0 / (1 + dist), d->nodes[0].beliefEuc[0], 9e-7);
    
    //manually calculate distance for node 0, centroid 3
    c1 = 0.95;
    dist = sqrt( (c2  - c1) * (c2 - c1) );
    assertFloatEquals( 1.0 / (1 + dist), d->nodes[0].beliefEuc[3], 6e-8);
    
    //manually calculate distance for node 3, centroid 0
    c1 = 0.05;
    c2 = 0.99;
    dist = sqrt( (c2  - c1) * (c2 - c1));
    assertFloatEquals( 1.0 / (1 + dist), d->nodes[3].beliefEuc[0], 2e-8);
    
    //manually calculate distance for node 3, centroid 3
    c1 = 0.95;
    c2 = 0.99;
    dist = sqrt( (c2  - c1) * (c2 - c1));
    assertFloatEquals( 1.0 / (1 + dist), d->nodes[3].beliefEuc[3], 6.3e-8);

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


    assertIntArrayEqualsV(d->uf_winCounts[0], dc->centroids[0], 0, 2, 1, 1);
    assertFloatArrayEqualsE2DV(d->uf_avgDelta[0], 1e-5, dc->centroids[0], d->nodes[0].ns,
        0.0,                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //average delta for shared centroid 0
        ((0.11 - 0.06) + (0.22 - 0.06)) / 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ( 0.88 - 0.86 ),                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ( 0.99 - 0.95 ),                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);//average delta for shared centroid 3
    

    Uniform_ApplyDeltas(d, 0, d->uf_sigma[0]);
    Uniform_ApplyDeltas(d, 1, d->uf_sigma[1]);

    assertTrue(n[0].nCounts == NULL); //these are null in uniform destin

    //check that the shared centroids were moved to the correct positions
    assertFloatArrayEqualsE2DV(n[0].mu, 1e-5, 4, 9,
        0.05,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 0, unchanged because it wasn't a winner
        0.165, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 1, averaged because two nodes picked it
                                                                   //averaged between node 0 and node 1 observations, .11 and .22 = .165
        0.88,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //moved directly to node 2 observation because only node 2 picked it
        0.99,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);//moved directly to node 3 observation because only node 3 picked it
    //Uniform_UpdateStarvation(d->nodes, 0);

    float ** ad = d->uf_avgDelta[0];
    //calculate muSqDiff for layer 0
    float msq = ad[0][0] * ad[0][0] + ad[1][0] * ad[1][0] + ad[2][0] * ad[2][0] + ad[3][0] * ad[3][0];
    assertFloatEquals(msq, n[0].muSqDiff, 4.5e-10);
    DestroyConfig(dc);
    DestroyDestin(d);
    return 0;
}

//same setup as testUniform, but call the main FormulateBelief function to make sure it calls everything in the correct order.
int testUniformFormulate(){

    DestinConfig * dc = CreateDefaultConfig(2);
    dc->inputDim = 1;
    dc->lambdaCoeff = 1;
    dc->gamma = 1;
    dc->starvCoeff = 0.1;
    assignFloatArray(dc->temperatures, 2, 10.0, 10.0);
    assignUIntArray(dc->centroids, 2, 4, 4); //4 shared centroids per layer

    Destin * d = InitDestinWithConfig(dc);

    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1; //turn on cluster training
    d->layerMask[1] = 1;

    float image []  = {.11,.22,.88,.99};//1 pixel for each of the 4 bottom layer nodes

    Node * n = &d->nodes[0];

    //set centroid locations
    //mu is a table nb x ns. ns = ni + nb + np + nc
    //nb = 4 (centroids), ns = 9
    //all nodes point to the same centroids in a layer for uniform destin
    assignFloatArray2D(n->mu, 4, 9,
        0.05, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.06, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.86, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
        0.95, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);

    FormulateBelief(d, image);

    //check that the shared centroids were moved to the correct positions
    assertFloatArrayEqualsE2DV(n->mu, 2e-8, 4, 9,
        0.05,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 0, unchanged because it wasn't a winner
        0.165, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //centroid location 1, averaged because two nodes picked it
                                                                   //averaged between node 0 and node 1 observations, .11 and .22 = .165
        0.88,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, //moved directly to node 2 observation because only node 2 picked it
        0.99,  0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25);//moved directly to node 3 observation because only node 3 picked it

    assertFloatArrayEqualsEV(d->uf_starv[0], 1e-12, 4,
        1.0 - dc->starvCoeff, 1.0, 1.0, 1.0);

    assertFloatEquals(d->muSumSqDiff, n[0].muSqDiff + n[4].muSqDiff, 4.5e-8 );

    // check that outputBeliefs are copied
    uint i;
    for (i = 0; i < d->nNodes; i++)
    {
        n = &(d->nodes[i]);
        assertFloatArrayEquals(n->belief, n->outputBelief, n->nb);
    }

    DestroyConfig(dc);
    DestroyDestin(d);
    return 0;
}

int testSaveDestin1(){
    //test that SaveDestin and LoadDestin are working propertly
    DestinConfig * dc = CreateDefaultConfig(2);

    dc->nClasses = 6;
    dc->lambdaCoeff = 0.96;
    dc->gamma = 0.78;
    assignFloatArray(dc->temperatures, 2, 7.5, 8.5);
    dc->nMovements = 4;
    assignUIntArray(dc->centroids, 2, 3, 4);

    Destin * d = InitDestinWithConfig(dc);

    uint ns0 = dc->inputDim + dc->centroids[0]+ dc->centroids[1] + dc->nClasses;
    uint ns1 = 4*dc->centroids[0] + dc->centroids[1] + 0 + dc->nClasses;

    SetBeliefTransform(d, DST_BT_BOLTZ);
    d->layerMask[0] = 1;
    d->layerMask[1] = 1;

    SetLearningStrat(d, CLS_FIXED);
    assertTrue(ns0 == d->nodes[0].ns);
    assertTrue(ns1 == d->nodes[4].ns);
    d->isRecurrent = true;
    uint maxNs = d->maxNs;

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
    float ** uf_avgDelta[2];
    uf_avgDelta[0] = copyFloatDim2Array(d->uf_avgDelta[0], dc->centroids[0], ns0);
    uf_avgDelta[1] = copyFloatDim2Array(d->uf_avgDelta[1], dc->centroids[1], ns1);
    SaveDestin(d, "unit_test_destin.save");
    DestroyDestin(d);
    d = NULL;//must be set to NULL or LoadDestin will try to destroy an invalid pointer
    d = LoadDestin(d, "unit_test_destin.save");

    assertTrue(d->isRecurrent);
    assertTrue(d->centLearnStrat == CLS_FIXED);
    assertTrue(d->nLayers == 2);
    assertIntArrayEqualsV(d->nb, 2, 3, 4);
    assertTrue(d->nc == 6);
    Node * n = &d->nodes[0];

    assertTrue(n->ni == 16);
    assertFloatEquals(0.001, n->beta, 1e-10);
    assertFloatEquals(0.96, n->lambdaCoeff, 1e-07); //accuracy is not very good
    assertFloatEquals( 0.78, n->gamma, 3e-8);
    assertFloatArrayEqualsEV(d->temp, 1e-12, 2, 7.5, 8.5 );
    assertTrue(n->starvCoeff == dc->starvCoeff);
    assertTrue(d->nMovements == dc->nMovements);
    assertTrue(d->isUniform == dc->isUniform);

    assertIntArrayEqualsV(d->layerMask, 2, 0, 0); //TODO: layer mask is not saved,should be 1, 1
    assertIntArrayEqualsV(d->layerSize, 2, 4, 1);
    assertTrue(d->maxNb == 4);
    assertTrue(d->maxNs == maxNs);
    assertFloatEquals(0.0, d->muSumSqDiff, 0); //it currently resets to 0

    assertFloatArrayEqualsE2D(uf_avgDelta[0], d->uf_avgDelta[0], dc->centroids[0], ns0, 1e-36);
    assertFloatArrayEqualsE2D(uf_avgDelta[1], d->uf_avgDelta[1], dc->centroids[1], ns1, 1e-36);

    DestroyDestin(d);
    uint c;
    for (c = 0; c < dc->centroids[0]; c++){
        FREE(uf_avgDelta[0][c]);
    }
    for (c = 0; c < dc->centroids[1]; c++){
        FREE(uf_avgDelta[1][c]);
    }
    FREE(uf_avgDelta[0]);
    FREE(uf_avgDelta[1]);
    DestroyConfig(dc);
    return 0;
}

void turnOnMask(Destin * d){
    int i;
    for(i = 0 ; i < d->nLayers ;i++){
        d->layerMask[i] = 1;
    }
}


int _testSaveDestin2(bool isUniform, CentroidLearnStrat learningStrat, BeliefTransformEnum bt){
    //Test that SaveDestin and LoadDestin are working propertly.
    //This uses the strategy of checking that the belief outputs are the same
    //after loading a saved destin and repeating the same input image.

    uint i, j;

    DestinConfig * dc = CreateDefaultConfig(4);
    assignUIntArray(dc->centroids, 4, 3, 4, 2, 4);
    dc->nClasses = 6;
    dc->lambdaCoeff = 0.56;
    dc->gamma = 0.28;
    dc->nMovements = 4;
    assignFloatArray(dc->temperatures, 4, 3.5, 4.5, 5.0, 4.4);
    dc->isUniform = isUniform;
    dc->isRecurrent = true;

    Destin * d = InitDestinWithConfig(dc);
    DestroyConfig(dc);

    SetLearningStrat(d, learningStrat);
    SetBeliefTransform(d, bt);
    turnOnMask(d);

    //generate random images
    uint image_size = d->nci[0] * d->layerSize[0];
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

    //get beliefs for layers
    float beliefsLayer0[64 * 3];
    float beliefsLayer1[16 * 4];
    float beliefsLayer2[4 * 2];

    //mix it up some more
    uint iterations = 50;
    for(i = 0 ; i < iterations; i++){
        for(j = 0 ; j < nImages ; j++){
            FormulateBelief(d, images[j]);

            GetLayerBeliefs(d, 0, beliefsLayer0);
            GetLayerBeliefs(d, 1, beliefsLayer1);
            GetLayerBeliefs(d, 2, beliefsLayer2);

            assertNoNans(beliefsLayer0, 64 * 3); //test that non nans (float "not a number") are occuring
            assertNoNans(beliefsLayer1, 16 * 4);
            assertNoNans(beliefsLayer2, 4 * 2);
        }
    }

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

    float newBeliefsLayer0[64 * 3];
    float newBeliefsLayer1[16 * 4];
    float newBeliefsLayer2[4 * 2];
    GetLayerBeliefs(d, 0, newBeliefsLayer0);
    GetLayerBeliefs(d, 1, newBeliefsLayer1);
    GetLayerBeliefs(d, 2, newBeliefsLayer2);

    assertFloatArrayEquals(beliefsLayer0, newBeliefsLayer0, 64 * 3);
    assertFloatArrayEquals(beliefsLayer1, newBeliefsLayer1, 16 * 4);
    assertFloatArrayEquals(beliefsLayer2, newBeliefsLayer2, 4 * 2);

    DestroyDestin(d);

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
    assertIntArrayEqualsV(d->layerMaxNb, 3, 7, 8, 9);
    assertIntArrayEqualsV(d->nb, 3, 2, 4, 5);
    assertFloatArrayEqualsEV(d->temp, 1e-12, 3, 3.1, 3.2, 3.3 );
    assertFloatEquals(d->addCoeff, 4.3, 1e-6);
    assertTrue(d->nc == 8);
    assertTrue(d->nMovements == 7);
    assertTrue(d->nLayers == 3);
    assertTrue(d->nci[0] == 4);
    Node * n = &d->nodes[0];
    assertTrue(n->ni == 4);
    assertFloatEquals(0.002, n->beta, 1e-8);
    assertFloatEquals(0.1, n->lambdaCoeff, 1e-8);
    assertFloatEquals(0.2, n->gamma, 1e-8);
    assertFloatEquals(0.001, n->starvCoeff, 1e-8);
    assertTrue(d->beliefTransform == DST_BT_BOLTZ);
    assertTrue(d->isRecurrent);
    DestroyDestin(d);
    return 0;
}


//test GenerateInputFromBelief to make sure it doesn't crash
int _testGenerateInputFromBelief(bool isUniform){

    DestinConfig * dc = CreateDefaultConfig(4);
    assignUIntArray(dc->centroids, 4, 3, 4, 2, 2);
    assignFloatArray(dc->temperatures, 4, 7.5, 8.5, 4.0, 4.0);
    dc->isUniform = isUniform;
    Destin * d = InitDestinWithConfig(dc);
    DestroyConfig(dc);

    d->layerMask[0] = 1;
    d->layerMask[1] = 1;
    d->layerMask[2] = 1;
    d->layerMask[3] = 1;

    float *outFrame;
    MALLOC( outFrame, float, d->layerSize[0] * d->nci[0]);
    uint i, nImages = 4;
    float ** images = makeRandomImages(d->layerSize[0] * d->nci[0], nImages);

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

int test8Layers(){
    Destin * d = makeDestin(8);

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
    //check some nodes from layer 0 that they have the right input offsets to their input data
    
    Destin * d = makeDestin(3);
    assertIntEquals(2, GetNodeFromDestin(d, 0, 0, 2)->nIdx);
    assertIntEquals(11, GetNodeFromDestin(d, 0, 2, 3)->nIdx);
    assertIntEquals(19, GetNodeFromDestin(d, 1, 1, 1)->nIdx);
    assertIntEquals(20, GetNodeFromDestin(d, 2, 0, 0)->nIdx);
       
    assertIntArrayEqualsV(GetNodeFromDestin(d, 0, 0, 0)->inputOffsets,
                          16, 0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 0, 3, 2)->inputOffsets,
                          16, 200, 201, 202, 203, 216, 217, 218, 219, 232, 233, 234, 235, 248, 249, 250, 251);
    assertIntArrayEqualsV(GetNodeFromDestin(d, 0, 2, 1)->inputOffsets,
                          16, 132, 133, 134, 135, 148, 149, 150, 151, 164, 165, 166, 167, 180, 181, 182, 183);

    // check that input offsets for nodes from layers above 0 are set to NULL
    assertTrue(GetNodeFromDestin(d, 1, 0, 0)->inputOffsets == NULL);
    assertTrue(GetNodeFromDestin(d, 2, 0, 0)->inputOffsets == NULL);

    DestroyDestin(d);

    return 0;
}


int testLinkParentsToChildren(){
    // Test that parent nodes have the right children in their children pointers
    Destin * d = makeDestin(4);

    Node * parent = GetNodeFromDestin(d, 1, 0, 2);

    assertIntEquals(66, parent->nIdx);
    assertIntEquals(4, parent->children[0]->nIdx);
    assertIntEquals(5, parent->children[1]->nIdx);
    assertIntEquals(12, parent->children[2]->nIdx);
    assertIntEquals(13, parent->children[3]->nIdx);

    // Children have parent as their parent
    assertTrue(parent->children[0]->parents[0] == parent);
    assertTrue(parent->children[1]->parents[0] == parent);
    assertTrue(parent->children[2]->parents[0] == parent);
    assertTrue(parent->children[3]->parents[0] == parent);
    assertTrue(parent->children[3]->firstParent == parent);

    // Parent of top layer node is null
    Node * root = GetNodeFromDestin(d, 3, 0, 0);
    assertTrue(root->parents[0] == NULL);

    Node * child = GetNodeFromDestin(d, 1, 2, 1);
    parent = GetNodeFromDestin(d, 2, 1, 0);
    assertTrue (child->parents[0] == parent);
    assertTrue (child->firstParent == parent);
    assertTrue (parent->parents[0] == root);

    DestroyDestin(d);

    // Check another
    uint layerWidths[] = {4, 2, 1};
    uint inputDim = 9;
    d = makeDestinFromLayerCfg(3, inputDim, layerWidths);

    root = GetNodeFromDestin(d, 2, 0, 0);
    assertIntEquals(20, root->nIdx);
    parent = GetNodeFromDestin(d, 1, 1, 1);
    assertIntEquals(19, parent->nIdx);
    assertIntEquals(10, parent->children[0]->nIdx);
    assertIntEquals(14, parent->children[2]->nIdx);
    assertTrue (parent->parents[0] == root);

    Node * node = GetNodeFromDestin(d, 0, 2, 1);
    assertIntArrayEqualsV(node->inputOffsets, 9, 75, 76, 77, 87, 88, 89, 99, 100, 101);

    DestroyDestin(d);

    // Check another geometry
    inputDim = 1;
    uint layerWidths2[] = {12, 3, 1, 1};
    d = makeDestinFromLayerCfg(4, inputDim, layerWidths2);

    root = GetNodeFromDestin(d, 3, 0, 0);
    assertIntEquals(154, root->nIdx);
    node = GetNodeFromDestin(d, 2, 0, 0);
    assertIntEquals(153, node->nIdx);
    parent = GetNodeFromDestin(d, 1, 1, 2);
    assertIntEquals(149, parent->nIdx);
    assertIntEquals(56, parent->children[0]->nIdx);
    assertIntEquals(69, parent->children[5]->nIdx);
    assertTrue(parent->children[0]->parents[0] == parent);
    assertTrue(parent->children[5]->parents[0] == parent);

    DestroyDestin(d);

    return 0;
}

int testCentroidImageGeneration(){
    DestinConfig * dc = CreateDefaultConfig(3); // create a 3 layer heiaracy
    dc->inputDim = 1; // 1 pixel input per node
    assignUIntArray(dc->centroids, 3, 2, 2, 2); // centroids per node, for each layer botttom to top.
    assignFloatArray(dc->temperatures, 3, 7.5, 8.5, 4.0); // arbitrary, doesn't matter

    Destin * d = InitDestinWithConfig(dc);

    DestroyConfig(dc);

    SetBeliefTransform(d, DST_BT_BOLTZ);

    Node * n = GetNodeFromDestin(d, 0, 0 ,0);
    n->mu[0][0] = 0.0; // assign centroid 0 white
    n->mu[1][0] = 1.0; // assign centroid 1 black ( or is it white?)

    n = GetNodeFromDestin(d, 1, 0 ,0);
    assignFloatArray(n->mu[0], 8,  0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); //1st centroid is a black 2x2 square
    assignFloatArray(n->mu[1], 8,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0); //2nd centroid is a white 2x2 square

    n = GetNodeFromDestin(d, 2, 0 ,0);

    //Assign 1st centroid to be top half black, bottom half white
    assignFloatArray(n->mu[0], 8,  1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0);

    //Assign 2nd centroid to be all black, bottom right dark grey
    assignFloatArray(n->mu[1], 8,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.75, 0.25);

    float **** images = Cig_CreateCentroidImages(d, 1.0);

    // Check that the generated images are correct.
    // Bottom centroid
    assertFloatArrayEqualsEV(images[0][0][0], 0.0, 1, 0.0); // 1st centroid is white
    assertFloatArrayEqualsEV(images[0][0][1], 0.0, 1, 1.0); // 2nd is black ( or is it the other way around...)

    assertFloatArrayEqualsEV(images[0][1][0], 0.0, 4, 1.0, 1.0, 1.0, 1.0); // 1st centroid is a black 2x2 square
    assertFloatArrayEqualsEV(images[0][1][1], 0.0, 4, 0.0, 0.0, 0.0, 0.0); // 2nd centroid is a white 2x2 square

    // The images array is indexed like this: images[channel][layer][centroid].
    // First centroid of top layer should generate an image of top half black
    // and bottom half white
    assertFloatArrayEqualsEV(images[0][2][0], 0.0, 16,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0,
                             0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0);

    // Second centroid of top layer should generate everything black (1.0)
    // except for the bottom right corner which is gray (0.75)
    assertFloatArrayEqualsEV(images[0][2][1], 0.0, 16,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 0.75, 0.75,
                             1.0, 1.0, 0.75, 0.75);

    // Test Cig_PowerNormalize that it normalizes vector [1,2,3] properly ( when exponent parameter is 2)
    float aDist[3] = {1.0,2.0,3.0};
    float aDistNormed[3];
    Cig_PowerNormalize(aDist, aDistNormed, 3, 2);
    assertFloatArrayEqualsEV(aDistNormed, 1e-6, 3, 0.0714285714, 0.2857142857, 0.6428571429 );
    //Try if the source is same as dest
    Cig_PowerNormalize(aDist, aDist, 3, 2);
    assertFloatArrayEqualsEV(aDist, 1e-6, 3, 0.0714285714, 0.2857142857, 0.6428571429 );

    Cig_DestroyCentroidImages(d, images);
    DestroyDestin(d);

    return 0;
}

int testColorCentroidImageGeneration(){
    DestinConfig * dc = CreateDefaultConfig(3); // create a 3 layer heiaracy
    dc->inputDim = 1; // 1 pixel input per node
    dc->isRecurrent = false;
    assignUIntArray(dc->centroids, 3, 3, 4, 2); // centroids per node, for each layer botttom to top.

    dc->extRatio = 3; // allow for RGB processing

    Destin * d = InitDestinWithConfig(dc);
    DestroyConfig(dc);

    Node * n = GetNodeFromDestin(d, 0, 0 ,0);

    assertIntEquals(10, n->ns);
    assertIntEquals(3, n->nb);

    // Assign bottom layer centroids.

    // First centroid represent white pixel ( Red + Green + Blue mixed).
    assignFloatArray(n->mu[0], n->ns,
            1.0, // Red
            .33, .33, .33, // previous self belielfs, this node has 3 centroids, one for each centroid
            .25, .25, .25, .25,// parent prev belief, parent nodes have 4 centroids
            /*empty*/ // class, empty because nc = 0
            1.0,// Green
            1.0 // Blue
            );

    // Second centroid represents a red pixel.
    assignFloatArray(n->mu[1],  n->ns,
            1.0, // Red
            .33, .33, .33, // previous self belielfs, this node has 3 centroids, one for each centroid
            .25, .25, .25, .25,// parent prev belief, parent nodes have 4 centroids
            /*empty*/ // class, empty because nc = 0
            0.0,// Green
            0.0 // Blue
            );


    // Third centroid represents a blue pixel.
    assignFloatArray(n->mu[2],  n->ns,
            0.0, // Red
            .33, .33, .33, // previous self belielfs, this node has 3 centroids, one for each centroid
            .25, .25, .25, .25,// parent prev belief, parent nodes have 4 centroids
            /*empty*/ // class, empty because nc = 0
            0.0,// Green
            1.0 // Blue
            );

    //// Assign layer 1 centrois
    n = GetNodeFromDestin(d, 1, 0 ,0);
    assertIntEquals(18, n->ns);
    // First centroid represent magenta 2x2 patch ( Red + Blue mixed).
    assignFloatArray(n->mu[0], n->ns,
            //W    R    B
            0.0, 0.5, 0.5, // 1st child node belief distribution, W R B, R+B = magenta
            0.0, 0.5, 0.5, // 2nd child
            0.0, 0.5, 0.5, // 3rd child
            0.0, 0.5, 0.5, // 4th child
            .25, .25, .25, .25, // previous self belielfs, this node has 4 centroids, one for each centroid
            .5, .5// parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    // Second centroid represent red 2x2 patch.
    assignFloatArray(n->mu[1], n->ns,
          //W    R    B
            0.0, 1.0, 0.0, // 1st child node belief distribution, W R B
            0.0, 1.0, 0.0, // 2nd child
            0.0, 1.0, 0.0, // 3rd child
            0.0, 1.0, 0.0, // 4th child
            .25, .25, .25, .25, // previous self belielfs, this node has 4 centroids, one for each centroid
            .5, .5// parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    // Third centroid represent Blue 2x2 patch.
    assignFloatArray(n->mu[2], n->ns,
          //W    R    B
            0.0, 0.0, 1.0, // 1st child node belief distribution, W R B
            0.0, 0.0, 1.0, // 2nd child
            0.0, 0.0, 1.0, // 3rd child
            0.0, 0.0, 1.0, // 4th child
            .25, .25, .25, .25, // previous self belielfs, this node has 4 centroids, one for each centroid
            .5, .5// parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    // Fourth centroid represent White 2x2 patch.
    assignFloatArray(n->mu[3], n->ns,
          //W    R    B
            1.0, 0.0, 0.0, // 1st child node belief distribution, W R B
            1.0, 0.0, 0.0, // 2nd child
            1.0, 0.0, 0.0, // 3rd child
            1.0, 0.0, 0.0, // 4th child
            .25, .25, .25, .25, // previous self belielfs, this node has 4 centroids, one for each centroid
            .5, .5// parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    //// Assign top layer centroids
    n = GetNodeFromDestin(d, 2, 0 ,0); // get top layer node
    assertIntEquals(18, n->ns);

    // First centroid represents: ( M = magenta, R = red, B = Blue, W = white)
    // MMRR
    // MMRR
    // BBWW
    // BBWW
    assignFloatArray(n->mu[0], n->ns,
          //M    R    B     W
            1.0, 0.0, 0.0, 0.0,// 1st child node belief distribution over centroids M R B W
            0.0, 1.0, 0.0, 0.0,// 2nd child
            0.0, 0.0, 1.0, 0.0, // 3rd child
            0.0, 0.0, 0.0, 1.0,// 4th child
            .5, .5 // previous self belielfs, this node has 2 centroids, one for each centroid
            /*empty no parent */ // parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    // Second centroid represents: ( M = magenta, R = red, B = Blue, W = white)
    // WWWW
    // WWWW
    // WWMM
    // WWMM
    assignFloatArray(n->mu[1], n->ns,
          //M    R    B     W
            0.0, 0.0, 0.0, 1.0,// 1st child node belief distribution over centroids M R B W
            0.0, 0.0, 0.0, 1.0,// 2nd child
            0.0, 0.0, 0.0, 1.0,// 3rd child
            1.0, 0.0, 0.0, 0.0,// 4th child
            .5, .5 // previous self belielfs, this node has 2 centroids, one for each centroid
            /*empty no parent */ // parent prev belief, parent nodes have 2 centroids
            /*empty*/ // class, empty because nc = 0
            /*empty*/ // extended input empty, because it only concerns the input layer
            );

    float **** images = Cig_CreateCentroidImages(d, 1.0);


    int channel = 0; // red channel
    // Check first centroid image for:
    // MMRR
    // MMRR
    // BBWW
    // BBWW
    // Check red channel
    //                      images[channel][layer][centoid]
    assertFloatArrayEqualsEV(images[channel][2][0], 1e-12, 16,
            0.5, 0.5, 1.0, 1.0,
            0.5, 0.5, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0);

    // green channel
    channel = 1;
    assertFloatArrayEqualsEV(images[channel][2][0], 1e-12, 16,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 1.0,
            0.0, 0.0, 1.0, 1.0);

    // blue channel
    channel = 2;
    assertFloatArrayEqualsEV(images[channel][2][0], 1e-12, 16,
            0.5, 0.5, 0.0, 0.0,
            0.5, 0.5, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0);

    // Check second centroid image for:
    // WWWW
    // WWWW
    // WWMM
    // WWMM

    // red channel
    channel = 0;
    assertFloatArrayEqualsEV(images[channel][2][1], 1e-12, 16,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.5, 0.5,
            1.0, 1.0, 0.5, 0.5);

    // green channel
    channel = 1;
    assertFloatArrayEqualsEV(images[channel][2][1], 1e-12, 16,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0);

    // blue channel
    channel = 2;
    assertFloatArrayEqualsEV(images[channel][2][1], 1e-12, 16,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 0.5, 0.5,
            1.0, 1.0, 0.5, 0.5);

    Cig_DestroyCentroidImages(d, images);
    DestroyDestin(d);
    return 0;
}

// Used for delete element callback tests
long _testDeleteLongCounter;
void _testDeleteLong(void * elem)
{
    _testDeleteLongCounter += *((long *) elem);
}

int testArrayOperations() {
    assertIntEquals(16, SIZEV(0));
    assertIntEquals(16, SIZEV(1));
    assertIntEquals(16, SIZEV(7));
    assertIntEquals(16, SIZEV(16));
    assertIntEquals(32, SIZEV(17));
    assertIntEquals(32, SIZEV(25));
    assertIntEquals(32, SIZEV(31));
    assertIntEquals(32, SIZEV(32));
    assertIntEquals(64, SIZEV(33));
    assertIntEquals(256, SIZEV(157));
    assertIntEquals(2048, SIZEV(1345));
    assertIntEquals(2097152, SIZEV(1345678));

    // check if writing at the end of the buffer does not cause memory crash
    float * array;

    MALLOCV(array, float, 5);         // should allocate 32 bytes (first power of 2 larger then 5*sizeof(float))
    array[6] = 1;                     // write to 24 byte offset (larger then 5*sizeof(float)
    array[7] = 2;                     // write again
    REALLOCV(array, float, 5, 2);     // should not resize the array
    array[6] = 1;
    REALLOCV(array, float, 7, 70);    // check if the array expands now
    array[75] = 1;
    assertFloatEquals(1.0, array[6], 1e-12); // check if the reallocation preserves data
    assertFloatEquals(2.0, array[7], 1e-12);

    REALLOCV(array, float, 77, -70);  // check if the array does not shrink
    array[75] = 1;
    REALLOCV(array, float, 7, 10);    // the array shrinks now (this is a side effect on purpose)
                                      // change size from 128 to 32 (* sizeof(float))
    REALLOCV(array, float, 17,1345678);  // check large array
    array[1097152] = 1;
    FREE(array);

    int * intArray;
    MALLOCV(intArray, int, 129);
    intArray[255] = 1;
    FREE(intArray);

    // Check array insert element
    MALLOCV(intArray,int,0);
    ArrayAppendInt(&intArray, 0, 3);
    ArrayAppendInt(&intArray, 1, 2);
    ArrayAppendInt(&intArray, 2, 1);
    ArrayInsertInt(&intArray, 3, 2, 4);
    ArrayInsertInt(&intArray, 4, 2, 5);
    assertIntArrayEqualsV(intArray, 5, 3, 2, 5, 4, 1);

    // Check array insert multiple
    int index[3] = {0, 2, 5};
    int values[3] = {11, 12, 13};
    ArrayInsertInts(&intArray, 5, index, values, 3);
    assertIntArrayEqualsV(intArray, 8, 11, 3, 2, 12, 5, 4, 1, 13);
    int index1[4] = {6, 6, 8, 8};
    int values1[4] = {21, 22, 23, 24};
    ArrayInsertInts(&intArray, 8, index1, values1, 4);
    assertIntArrayEqualsV(intArray, 12, 11, 3, 2, 12, 5, 4, 21, 22, 1, 13, 23, 24);

    // Check array delete
    ArrayDeleteInt(&intArray, 12, 11);
    ArrayDeleteInt(&intArray, 11, 6);
    ArrayDeleteInt(&intArray, 10, 6);
    ArrayDeleteInt(&intArray, 9, 0);
    assertIntArrayEqualsV(intArray, 8, 3, 2, 12, 5, 4, 1, 13, 23);

    // Check array delete multiple
    int index2[5] = {0, 3, 4, 6, 7};
    ArrayDeleteInts(&intArray, 8, index2, 5);
    assertIntArrayEqualsV(intArray, 3, 2, 12, 1);
    FREE(intArray);

    // Check other types
    int ** intPointerArray;
    MALLOCV(intPointerArray,int *,0);
    int * intPointerValues[3] = {index, index1, index2};
    int index3[3] = {0, 0, 0};
    ArrayInsertPtrs((void *) &intPointerArray, 0, index3, intPointerValues, 3);
    assertTrue(intPointerArray[2] == index2);
    ArrayDeletePtr((void *) &intPointerArray, 3, 1);
    assertTrue(intPointerArray[1] == index2);
    FREE(intPointerArray);

    // Check if delete element callback is executed
    long * longArray;
    MALLOCV(longArray, long, 0);
    ArrayAppendLong(&longArray, 0, 5);
    int index4[3] = {1, 1, 1};
    long values4[3] = {7, 8, 9};
    ArrayInsertLongs(&longArray, 1, index4, values4, 3);
    assertLongArrayEqualsV(longArray, 4, 5L, 7L, 8L, 9L);

    _testDeleteLongCounter = 0;
    ArrayDeleteElement((void *)&longArray, sizeof(long), 4, 2, &_testDeleteLong);
    assertTrue(_testDeleteLongCounter == 8L);
    int index5[2] = {0, 2};
    ArrayDeleteMultiple((void *)&longArray, sizeof(long), 3, index5, 2, &_testDeleteLong);
    assertTrue(_testDeleteLongCounter == 22L);
    FREE(longArray);

    // Check delete array
    float * floatArray1;
    float * floatArray2;
    MALLOCV(floatArray1, float, 100);
    MALLOCV(floatArray2, float, 100);
    float ** floatArrayArray;
    MALLOCV(floatArrayArray, float *, 0);
    ArrayAppendPtr((void *)&floatArrayArray, 0, floatArray1);
    ArrayAppendPtr((void *)&floatArrayArray, 1, floatArray2);
    int index6[2] = {0, 1};
    ArrayDeleteArrays((void *)&floatArrayArray, 2, index6, 2);   // shoud free internal arrays
    FREE(floatArrayArray);
    return 0;
}

int testBuildOverlappingHeirarchy(){
    // Makes a heirarchy that has both overlapping and non overlapping nodes.
    // Parent layer has overlapping nodes if its width is one less than the child layer width.
    // Otherwise the parent layer width must divide evenly into the child layer.

    DestinConfig * config = CreateDefaultConfig(5);
    assertTrue(config->inputDim == 16);

    // Specify the heirarchy with layerWidths
    assignUIntArray(config->layerWidths, 5,
                    8, 7, 6, 2, 1);

    Destin * d = InitDestinWithConfig(config);

    assertIntEquals(4, GetNodeFromDestin(d, 1, 0, 0)->nChildren);

    assertIntEquals(9, GetNodeFromDestin(d, 3, 0, 0)->nChildren);

    // make sure it transfered over
    assertUIntArrayEqualsV(d->layerWidth, 5, 8, 7, 6, 2, 1);

    // Check that the children input counts were computed correctly from the layer widths
    assertUIntArrayEqualsV(d->nci, 5, 16, 4, 4, 9, 4);

    // Check that the bottom layer nodes have the correct number of parents
    // for the overlapping node regions.
    // Child nodes that are overlapping each can have up to 4 parents.
    uint bottomExpectedParentCounts [] = {1, 2, 2, 2, 2, 2, 2, 1,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          2, 4, 4, 4, 4, 4, 4, 2,
                                          1, 2, 2, 2, 2, 2, 2, 1};
    int i;
    for(i = 0 ; i < 8*8 ; i++){
        assertIntEquals(bottomExpectedParentCounts[i], GetNodeFromDestinI(d, 0, i)->nParents);
    }

    for(i = 0 ; i < 6*6 ; i++){
        assertIntEquals(1, GetNodeFromDestinI(d, 2, i)->nParents);
    }


    // Check that the top node has no parents
    assertTrue(GetNodeFromDestin(d, 4, 0, 0)->nParents == 0);
    assertTrue(GetNodeFromDestin(d, 4, 0, 0)->firstParent == NULL);

    // Check that overlapping child nodes have the rigt parents in the right positions.
    // Child nodes that are overlapping each can have up to 4 parents.
    Node *childNode = GetNodeFromDestin(d, 1, 0, 0);
    assertTrue(childNode->nParents == 1);
    assertTrue(childNode->parents[0] == NULL); // North West parent
    assertTrue(childNode->parents[1] == NULL); // North East parent
    assertTrue(childNode->parents[2] == NULL); // South West parent
    assertTrue(childNode->parents[3] == GetNodeFromDestin(d, 2, 0, 0)); // South East parent
    assertTrue(childNode->firstParent == childNode->parents[3]);

    childNode = GetNodeFromDestin(d, 1, 0, 1);
    assertTrue(childNode->nParents == 2);
    assertTrue(childNode->parents[0] == NULL); // North West parent
    assertTrue(childNode->parents[1] == NULL); // North East parent
    assertTrue(childNode->parents[2] == GetNodeFromDestin(d, 2, 0, 0)); // South West parent
    assertTrue(childNode->parents[3] == GetNodeFromDestin(d, 2, 0, 1)); // South East parent
    assertTrue(childNode->firstParent == childNode->parents[2]);

    childNode = GetNodeFromDestin(d, 1, 1, 3);
    assertTrue(childNode->nParents == 4);
    assertTrue(childNode->parents[0] == GetNodeFromDestin(d, 2, 0, 2)); // North West parent
    assertTrue(childNode->parents[1] == GetNodeFromDestin(d, 2, 0, 3)); // North East parent
    assertTrue(childNode->parents[2] == GetNodeFromDestin(d, 2, 1, 2)); // South West parent
    assertTrue(childNode->parents[3] == GetNodeFromDestin(d, 2, 1, 3)); // South East parent
    assertTrue(childNode->firstParent == childNode->parents[0]);

    // Check that a parent node has the right children
    Node *parentNode = GetNodeFromDestin(d, 2, 1, 4);
    assertTrue(parentNode->children[0] == GetNodeFromDestin(d, 1, 1, 4));
    assertTrue(parentNode->children[1] == GetNodeFromDestin(d, 1, 1, 5));
    assertTrue(parentNode->children[2] == GetNodeFromDestin(d, 1, 2, 4));
    assertTrue(parentNode->children[3] == GetNodeFromDestin(d, 1, 2, 5));
    assertTrue(childNode->firstParent == childNode->parents[0]);

    assertTrue(GetNodeFromDestin(d, 1, 1, 4)->parents[3] == parentNode);
    assertTrue(GetNodeFromDestin(d, 1, 1, 5)->parents[2] == parentNode);
    assertTrue(GetNodeFromDestin(d, 1, 2, 4)->parents[1] == parentNode);
    assertTrue(GetNodeFromDestin(d, 1, 2, 5)->parents[0] == parentNode);

    DestroyDestin(d);
    DestroyConfig(config);
    return 0;
}

int testExtRatio()
{
    // extRatio will affect 'ns', thus the size of 'observation', 'mu' and the related parameters
    DestinConfig * dc = CreateDefaultConfig(1);
    int inputDim = 16;
    dc->inputDim = inputDim;
    uint nb_0 = 4;
    dc->centroids[0] = nb_0;
    dc->extRatio = 3;
    dc->layerMaxNb[0] = 4;

    Destin * d = InitDestinWithConfig(dc);

    DestroyConfig(dc);

    float image[48] = {
        .01, .02, .03, .04,
        .05, .06, .07, .08,
        .09, .10, .11, .12,
        .13, .14, .15, .16,
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .9, .9, .9, .9,
        .9, .9, .9, .9,
        .9, .9, .9, .9,
        .9, .9, .9, .9
    };

    Node * n = &d->nodes[0];

    assertTrue(n->ni == 16);
    int extRatio = 3;
    assertTrue(n->d->extRatio == extRatio);
    assertTrue(n->ns == inputDim * extRatio + nb_0 + 0 + 0 /*nc*/);

    // GetObservation; test whether it's extended to contain more info;
    int nid = 0;
    GetObservation( d->nodes, image, nid );
    float expected_obs [] = {
        .01, .02, .03, .04,
        .05, .06, .07, .08,
        .09, .10, .11, .12,
        .13, .14, .15, .16,
        0.25, 0.25, 0.25, 0.25, // self previous beliefs, no parent previous beliefs
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .5, .5, .5, .5,
        .9, .9, .9, .9,
        .9, .9, .9, .9,
        .9, .9, .9, .9,
        .9, .9, .9, .9
    };
    assertFloatArrayEquals( expected_obs , n->observation, 52);
    DestroyDestin(d);
    return 0;
}

int main(int argc, char ** argv ){

    //RUN( shouldFail );
    RUN(testVarArgs);

    RUN(testInit);
    RUN(testInputOffsets);
    RUN(testFormulateNotCrash);
    RUN(testFormulateStages);
    RUN(testLinkParentsToChildren);
    RUN(testUniform);
    RUN(testUniformFormulate);
    RUN(testSaveDestin1);
    RUN(testSaveDestin2);
    RUN(testLoadFromConfig);
    RUN(testGenerateInputFromBelief);
    RUN(test8Layers);
    RUN(testArrayOperations);
    RUN(testCentroidImageGeneration);
    RUN(testColorCentroidImageGeneration);
    RUN(testBuildOverlappingHeirarchy);
    RUN(testExtRatio);

    UT_REPORT_RESULTS();

    return 0;
}

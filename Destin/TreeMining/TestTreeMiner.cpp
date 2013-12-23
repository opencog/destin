
#include <vector>
#include "unit_test.h"
#include "CMOrderedTreeMinerWrapper.h"
#include "DestinNetworkAlt.h"
#include "VideoSource.h"
#include "DestinTreeManager.h"

using std::vector;

/*
 * This file contains many unit tests.
 * Each unit test is run with the RUN() macro in the main function.
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

int testTreeToVector(){

    // format is as follows:
    // id id <length of depth first search path> < the depth first search path>
    // Note that -1 means a back track in the dfs.
    stringstream ss("0 0 9 1 2 3 -1 4 -1 -1 5 -1");

    TextTree tt;
    ss >> tt;

    CMOrderedTreeMinerWrapper tmw;
    vector<short> v;
    tmw.treeToVector(tt, v);

    assertIntEquals(9, v.size());
    assertShortArrayEqualsV(v.data(), 9,
                            1, 2, 3, -1, 4, -1, -1, 5, -1);

    return 0;
}

int testTreeMiner(){
    CMOrderedTreeMinerWrapper tmw;

    short t1[9] = {1, 2, 3, -1, 4, -1, -1, 5, -1};
    short t2[5] = {1, 2, -1, 5, -1};

    tmw.addTree(t1, 9);
    tmw.addTree(t2, 5);

    vector<PatternTree> maximal_subtrees;

    const int support = 2;
    tmw.mine(support, maximal_subtrees);

    assertIntEquals(1, maximal_subtrees.size());

    vector<short> tree_out;
    tmw.treeToVector(maximal_subtrees[0], tree_out);
    assertIntEquals(5, tree_out.size());
    assertShortArrayEqualsV(&tree_out[0], 5, 1, 2, -1, 5, -1);
    return 0;
}

int testDisplayTree(){
    uint centroids[3] = {2, 2, 2};
    DestinNetworkAlt dn(W16, 3, centroids, true);

    Node * n = dn.getNode(0,0,0);
    assertIntEquals(20, n->ns);

    /// set black and white for level 0
    for(int i = 0 ; i < n->ns - 4 ; i++){ // ns - 4 skips the recurrence and parent belief section of the centroids
        //n->mu is shared with all nodes in uniform destin
        n->mu[0][i] = 1.0; //centroid 0 is black block
        n->mu[1][i] = 0.0; //centroid 1 is white block
    }

    n = dn.getNode(1,0,0);
    assertIntEquals(12, n->ns );

    /// set centroids to black and white for level 1
    assignFloatArray(n->mu[0], 8,
                     1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0); //black

    assignFloatArray(n->mu[1], 8,
                     0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); //white


    /// set centroids to black and white for top level
    n = dn.getNode(2,0,0);
    assertIntEquals(10, n->ns );

    assignFloatArray(n->mu[0], 8,
                     1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0); //black

    assignFloatArray(n->mu[1], 8,
                     0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0); //white

    /****
    * make a tree that represents all black
    ******/
    DestinTreeManager tm(dn, 0);
    vector<short> tree;
    tree.push_back(tm.getTreeLabelForCentroid(0,2,0)); //all black

    cv::Mat img = tm.getTreeImg(tree);

    assertIntEquals(16, img.cols);
    assertIntEquals(16, img.rows);

    // make sure that it generates a totally black image
    float * data = (float *)img.data;
    for(int i = 0 ; i < 16*16; i++){
        assertFloatEquals(1.0, data[i], 0.0); //make sure pixel is black
    }


    /**********************************************
    * Now generate a tree that represents this 16x16 pixel image,
    * where # is a 4x4 black patch and 0 is a 4x4 white patch:
    *
    *  ####
    *  ####
    *  #000
    *  0#00
    *
    **************************************************/
    // TODO: check if 1000 is a limitation in the mining software

    tree.clear();
    tree.push_back(tm.getTreeLabelForCentroid(0,2,0));
    tree.push_back(tm.getTreeLabelForCentroid(0,1,2));
    tree.push_back(tm.getTreeLabelForCentroid(1,0,1));
    tree.push_back(-1);
    tree.push_back(tm.getTreeLabelForCentroid(1,0,2));
    tree.push_back(-1);
    tree.push_back(-1);
    tree.push_back(tm.getTreeLabelForCentroid(1,1,3));
    tree.push_back(-1);

    img = tm.getTreeImg(tree);
    assertIntEquals(16, img.cols);
    assertIntEquals(16, img.rows);

    data = (float *)img.data;

    /*****
      Checks that the above image is generated
    ******/
    // Top two rows ( 0 and 1) should be black
    // Represents ####
    //            ####
    for(int i = 0 ; i < 8 * 16; i++){
        assertFloatEquals(1.0, *data, 0.0); // black
        data++; //move pointer to next pixel
    }

    // row 2
    // Represents #000
    for(int i = 0 ; i < 4; i++){ //4 horizonal scan lines per row

        //first cell is black
        for(int j = 0 ; j < 1 * 4; j++){
            assertFloatEquals(1.0, *data, 0.0); //pixel must be black
            data++; //move pointer to next pixel
        }

        // next 3 are white
        for(int j = 0 ; j < 3 * 4; j++){
            assertFloatEquals(0.0, *data, 0.0); //pixel must be white
            data++; //move pointer to next pixel
        }

    }

    // row 4
    // Represents 0#00
    for(int i = 0 ; i < 4; i++){ //4 horizonal scan lines per row

        //first cell is white
        for(int j = 0 ; j < 1 * 4; j++){
            assertFloatEquals(0.0, *data, 0.0); //pixel must be white
            data++; //move pointer to next pixel
        }

        // second cell is black
        for(int j = 0 ; j < 1 * 4; j++){
            assertFloatEquals(1.0, *data, 0.0); //pixel must be black
            data++; //move pointer to next pixel
        }

        // 3rd and 4th are white
        for(int j = 0 ; j < 2 * 4; j++){
            assertFloatEquals(0.0, *data, 0.0); //pixel must be white
            data++; //move pointer to next pixel
        }

    }

    return 0;
}

int testBeliefExporter(){
    uint centroids[3] = {2, 2, 2};
    DestinNetworkAlt dn(W16, 3, centroids, true);

    Destin * d = dn.getNetwork();
    for(int n = 0 ; n < d->nNodes ; n++){
        d->nodes[n].winner = n; // pretend winning centroid is n
                                // ( even though each node currenly only has 2 centroid)
    }
    DestinTreeManager be(dn, 0);

    int len = 41;// length of the depth first search path for a 3 layer heirach
    int b0 = 0;

    /// 32768  is the max value of short, the subtree mining software uses shorts values as node labels.
    /// This arithmetic mess below shows how I encode the ( layer#, centroid#, child position ) tupple into one short value.

    int b1 = ( 32768 / 3 ) * 1; // divide labels up into 3 layers
    int b2 = ( 32768 / 3 ) * 2;

    int cbs  = ( 32768 / 3 ) / 4;//child bucket size, divides each layer number range into 4 child position buckets

    int c00 = b0 + 0, c01 = b0 + cbs * 1, c02 = b0 + cbs * 2, c03 = b0 + cbs * 3; // children 0 to 3 offsets for level 0 ( bottom layer )
    int c10 = b1 + 0, c11 = b1 + cbs * 1, c12 = b1 + cbs * 2, c13 = b1 + cbs * 3; //                             level 1 ( middle layer )
    int c20 = b2 + 0, c21 = b2 + cbs * 1, c22 = b2 + cbs * 2, c23 = b2 + cbs * 3; //                             level 2 ( top layer )


    assertIntEquals(len, be.getWinningCentroidTreeSize());

    printf("b1: %i, b2: %i\n", b1, b2);
    short * t = be.getWinningCentroidTree();
    for(int i = 0 ; i < len; i++){
        printf("%i ",t[i]);
    }
    printf("\n");

    ///Create a representative depth first search tree path using node labels based on the winning centroid
    assertShortArrayEqualsV(be.getWinningCentroidTree(), len,
        c20 + 20, // "this "c20 + 20" means centroid #20 on level 2 in child position zero ( goes zero to three)
        c10 + 16, c00 + 0,  -1, c01 + 1,  -1, c02 + 4,  -1, c03 + 5,  -1, -1,
        c11 + 17, c00 + 2,  -1, c01 + 3,  -1, c02 + 6,  -1, c03 + 7,  -1, -1,
        c12 + 18, c00 + 8,  -1, c01 + 9,  -1, c02 + 12, -1, c03 + 13, -1, -1,
        c13 + 19, c00 + 10, -1, c01 + 11, -1, c02 + 14, -1, c03 + 15, -1, -1);

    return 0;
}

int experiment(){
    uint centroids[8] = {2,4,4,16,8,8,4,4};
    DestinNetworkAlt dn(W512, 8, centroids, true);
    dn.setBeliefTransform(DST_BT_P_NORM);
    float temperatures[8] = {2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};
    dn.setTemperatures(temperatures);
    dn.setIsPOSTraining(true);
    VideoSource vs(false, "moving_circle.avi");

#if FALSE
    for(int i = 200 ; i--; ){
        if(!vs.grab()){
            continue;
        }

        if(i % 10 == 0){
            printf("iteration: %i\n",i);
        }
        dn.doDestin(vs.getOutput());
    }
    dn.save("treeminer.dst");
#else
    dn.load("treeminer.dst");
#endif

    dn.setIsPOSTraining(false);
    DestinTreeManager be(dn, 0);
    be.setBottomLayer(5);
    CMOrderedTreeMinerWrapper tmw;


    for(int i = 50 ; i-- ;){
        if(!vs.grab()){
            printf("couldn't grab frame\n");
            continue;
        }
        // TODO: destin has 8 different frames in it at once,
        // due to the belief pipeline. maybe tree mining would
        // be more consistent if only 1 layer at a time was processed.

        dn.doDestin(vs.getOutput());
        int len = be.getWinningCentroidTreeSize();
        cout << "tree size is " << len << endl;
        cout << "getting tree " << i << endl;
        short * t = be.getWinningCentroidTree();

        for(int j = 0 ; j < len; j++){
            cout << t[j] << " ";
        }
        cout << endl;
        printf("adding tree %i\n ", i);
        tmw.addTree(t, len);
        printf("finished adding %i\n", i);
    }

    vector<PatternTree> minedTrees;
    int support = 10;
    printf("mining...\n");
    tmw.mine(support, minedTrees);
    vector<short> atree;
    tmw.treeToVector(minedTrees.at(7), atree);

    be.displayTree(atree);
    cv::waitKey();
    printf("trees: %lu\n", minedTrees.size());
    for(int i = 0 ; i < minedTrees.size(); i++){
        cout << minedTrees[i] << endl;
    }

    return 0;
}

void setWinningCentroids(Destin * dest, uint a, uint b, uint c, uint d){
    uint wc[4] = {a, b, c, d};
    for(int layer = 0 ; layer < 4; layer++){
        for(int i = 0 ; i < dest->layerSize[layer]; i++){
            GetNodeFromDestinI(dest, layer, i)->winner = wc[layer];
        }
    }
    return;
}

int testTimeSliceTreeExporter(){
    uint centroids[] = {4,4,4,4};
    DestinNetworkAlt dn(W32, 4, centroids, true);
    int bottom_layer = 1;
    DestinTreeManager tm(dn, bottom_layer);
    CMOrderedTreeMinerWrapper & tmw = tm.getTreeMiner();

    Destin * d = dn.getNetwork();

    setWinningCentroids(d, 1,2,3,4);
    tm.addTree();
    setWinningCentroids(d, 4,1,2,3);
    tm.addTree();
    setWinningCentroids(d, 3,4,1,2);
    tm.addTree();
    setWinningCentroids(d, 2,3,4,1);
    tm.addTree();
    setWinningCentroids(d, 1,2,3,4);
    tm.addTree();


    assertIntEquals(5, tm.getAddedTreeCount());
    tm.timeShiftTrees();

    assertIntEquals(3, tm.getAddedTreeCount());

    vector<short> tree;

    int treesize = ((1 + 4 + 16) * 2 - 1);

    tmw.treeToVector(tmw.getAddedTree(0), tree);
    assertIntEquals(treesize, tree.size());
    assertIntEquals(tm.getWinningCentroidTreeSize(), tree.size());

    setWinningCentroids(d, 2, 2, 2, 2); // bottom layer is 1, so it starts at 2
    assertShortArrayEquals(tm.getWinningCentroidTree(), tree.data(), tree.size() );

    tmw.treeToVector(tmw.getAddedTree(1), tree);
    setWinningCentroids(d, 1, 1, 1, 1);
    assertShortArrayEquals(tm.getWinningCentroidTree(), tree.data(), tree.size() );

    tmw.treeToVector(tmw.getAddedTree(2), tree);
    setWinningCentroids(d, 4, 4, 4, 4);
    assertShortArrayEquals(tm.getWinningCentroidTree(), tree.data(), tree.size() );

    return 0;
}

int testIsSubtreeOf(){

    CMOrderedTreeMinerWrapper tmw;

    short t1[1] = {5};
    tmw.addTree(t1, 1); // #0 - Add a one node tree.
    tmw.addTree(t1, 1); // #1 - Add a one node tree.


    t1[0] = 4;

    tmw.addTree(t1, 1); // #2 - Add a one node tree.

    short t2[5] = {5, 4, -1, 2, -1};
    tmw.addTree(t2, 5); // #3 - 3 nodes tree

    short t3[3] = {5, 2, -1};
    tmw.addTree(t3, 3); // #4 - 2 nodes tree

    assertTrue( tmw.isSubTreeOf(tmw.getAddedTree(0), tmw.getAddedTree(0)));// a tree appears in itself
    assertTrue( tmw.isSubTreeOf(tmw.getAddedTree(0), tmw.getAddedTree(1)));// a tree appears in itself
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(0), tmw.getAddedTree(2))) // does not appear in itself

    assertTrue( tmw.isSubTreeOf(tmw.getAddedTree(3), tmw.getAddedTree(4)));//tree #4 is found in tree #3
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(4), tmw.getAddedTree(3)));//tree #3 is not found in tree #4

    short t4[7] = {2, 3, -1, 5, -1, 5, -1}; tmw.addTree(t4, 7); // #5
    short t5[3] = {2, 4, -1};             ; tmw.addTree(t5, 3); // #6

    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(5), tmw.getAddedTree(6)) );
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(6), tmw.getAddedTree(5)) );

    short t6[7] = {2, 3, -1, 4, -1, 5, -1}; tmw.addTree(t6, 7); // #7
    assertTrue (tmw.isSubTreeOf(tmw.getAddedTree(7), tmw.getAddedTree(6)) );
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(6), tmw.getAddedTree(7)) );

    short t7[17] = {0,1,3,-1,4,-1,5,-1,-1,2,6,-1,7,-1,8,-1-1}; tmw.addTree(t7, 17); // #8
    short t8[13] = {0,1,3,-1,5,-1,-1,2,6,-1,8,-1-1};           tmw.addTree(t8, 13); // #9

    assertTrue (tmw.isSubTreeOf(tmw.getAddedTree(8), tmw.getAddedTree(9)) );
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(9), tmw.getAddedTree(8)) );

    short t9[5] = {0,1,5,-1,-1}; tmw.addTree(t9, 5); // #10
    assertTrue (tmw.isSubTreeOf(tmw.getAddedTree(8), tmw.getAddedTree(10)) );


    short t10[5] = {2,6,-1,8,-1}; tmw.addTree(t10, 5); // #11

    assertIntEquals(2, tmw.getAddedTree(8).vLabel.at(5));
    assertTrue (tmw.treeMatchesHelper(tmw.getAddedTree(8), tmw.getAddedTree(11), 5 ) != -1);

    assertTrue (tmw.isSubTreeOf(tmw.getAddedTree(8), tmw.getAddedTree(11)) );


    short t11[7] = {1,3,-1,5,-1,6,-1}; tmw.addTree(t11, 7); // #12
    assertFalse(tmw.isSubTreeOf(tmw.getAddedTree(8), tmw.getAddedTree(12)) );

    return 0;
}

int testFindSubtreeLocations(){
    CMOrderedTreeMinerWrapper tmw;

    short t0[7] = {0,1,-1,2,-1,3,-1};   tmw.addTree(t0, 7); // #0
    short t1[1] = {4};                  tmw.addTree(t1, 1); // #1
    short t2[1] = {2};                  tmw.addTree(t2, 1); // #2
    short t3[1] = {2};                  tmw.addTree(t3, 1); // #3
    short t4[3] = {2,2,-1};             tmw.addTree(t4, 3); // #4

    short t5[17] = {0,1,2,3,-1,-1,-1,5,1,2,-1,2,7,-1,-1,-1,-1};
                                        tmw.addTree(t5, 17);// #5

    short t6[3] = {1,2,-1};             tmw.addTree(t6, 3); // #6
    short t7[3] = {0,5,-1};             tmw.addTree(t7, 3); // #7
    short t8[1] = {1};                  tmw.addTree(t8, 1); // #8

    short t9[25] = {0,1,3,-1,6,-1,8,-1,-1,0,1,3,7,-1,-1,8,-1,6,-1,-1,0,1,-1,-1,-1};
                                        tmw.addTree(t9,25);// #9

    short t10[5] = {1,3,-1,6,-1};       tmw.addTree(t10, 5);// #10
    short t11[3] = {0,1,-1};            tmw.addTree(t11, 3);// #11
    short t12[3] = {0,0,-1};            tmw.addTree(t12, 3);// #12

    // shouldn't find one node with label 4 in the bigger tree
    assertIntEquals(-1, tmw.findSubtreeLocation(tmw.getAddedTree(0), tmw.getAddedTree(1) ));

    vector<int> locations = tmw.findSubtreeLocations(tmw.getAddedTree(0), tmw.getAddedTree(1) );
    assertIntEquals(0, locations.size()); // should be no locations, so zero size.


    // One node tree matched on one node tree should fine one match location.
    locations = tmw.findSubtreeLocations(tmw.getAddedTree(2), tmw.getAddedTree(3) );
    assertIntEquals(1, locations.size());
    assertIntEquals(0, locations.at(0));

    // The one node tree should be found in the two locations.
    locations = tmw.findSubtreeLocations(tmw.getAddedTree(4), tmw.getAddedTree(2) );
    assertIntEquals(2, locations.size());
    assertIntEquals(0, locations.at(0));
    assertIntEquals(1, locations.at(1));

    // The one node tree should be found at node position 2.
    locations = tmw.findSubtreeLocations(tmw.getAddedTree(0), tmw.getAddedTree(2) );
    assertIntEquals(2, locations.at(0));
    assertIntEquals(1, locations.size());

    // The two node tree should be found at two positions in the large tree.
    assertTrue(tmw.isSubTreeOf(tmw.getAddedTree(5), tmw.getAddedTree(6)));
    locations = tmw.findSubtreeLocations(tmw.getAddedTree(5), tmw.getAddedTree(6) );
    assertIntEquals(2, locations.size());
    assertIntEquals(1, locations.at(0));
    assertIntEquals(5, locations.at(1));

    // The two node tree should be found at one position in the large tree.
    locations = tmw.findSubtreeLocations(tmw.getAddedTree(5), tmw.getAddedTree(7) );
    assertIntEquals(1, locations.size());
    assertIntEquals(0, locations.at(0));

    locations = tmw.findSubtreeLocations(tmw.getAddedTree(5), tmw.getAddedTree(2) );
    assertIntEquals(3, locations.size());
    assertIntEquals(2, locations.at(0));
    assertIntEquals(6, locations.at(1));
    assertIntEquals(7, locations.at(2));

    locations = tmw.findSubtreeLocations(tmw.getAddedTree(9), tmw.getAddedTree(10) );
    assertIntEquals(2, locations.size());
    assertIntEquals(1, locations.at(0));
    assertIntEquals(6, locations.at(1));

    locations = tmw.findSubtreeLocations(tmw.getAddedTree(9), tmw.getAddedTree(11) );
    assertIntEquals(3, locations.size());
    assertIntEquals(0, locations.at(0));
    assertIntEquals(5, locations.at(1));
    assertIntEquals(11, locations.at(2));

    locations = tmw.findSubtreeLocations(tmw.getAddedTree(9), tmw.getAddedTree(12) );
    assertIntEquals(2, locations.size());
    assertIntEquals(0, locations.at(0));
    assertIntEquals(5, locations.at(1));

    assertIntEquals(-1, tmw.findSubtreeLocation(tmw.getAddedTree(9), tmw.getAddedTree(6)));

    return 0;
}

int testOverLappingDestinHeirarchy(){
    uint centroids []= {4,4,4,4};
    uint layer_widths[] = {4,3,2,1};

    DestinNetworkAlt dn(W16, 4, centroids, true, layer_widths, DST_IMG_MODE_GRAYSCALE);
    DestinTreeManager dtm(dn, 0);

    assertIntEquals(169,dtm.getWinningCentroidTreeSize())
    float ** images = makeRandomImages(16*16, 2);
    dn.doDestin(images[0]);
    dtm.addTree();
    dn.doDestin(images[1]);
    dtm.addTree();
    freeRandomImages(images, 2);
    return 0;
}

int main(int argc, char ** argv){
    RUN(testTreeToVector);
    RUN(testTreeMiner);
    RUN(testBeliefExporter);
    //RUN(experiment); //commented out because it needs a video file
    RUN(testDisplayTree);
    RUN(testTimeSliceTreeExporter);
    RUN(testIsSubtreeOf);
    RUN(testFindSubtreeLocations);
    RUN(testOverLappingDestinHeirarchy);
    UT_REPORT_RESULTS();
    return 0;
}


#include <vector>
#include "unit_test.h"
#include "CMOrderedTreeMinerWrapper.h"
#include "DestinNetworkAlt.h"
#include "VideoSource.h"
#include "BeliefExporter.h"

using std::vector;

int testTreeToVector(){

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

    tmw.addTreeToDatabase(t1, 9);
    tmw.addTreeToDatabase(t2, 5);

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

int testBeliefExporter(){
    uint centroids[3] = {2, 2, 2};
    DestinNetworkAlt dn(W16, 3, centroids, true);

    Destin * d = dn.getNetwork();
    for(int n = 0 ; n < d->nNodes ; n++){
        d->nodes[n].winner = n; // pretend winning centroid is n
                                // ( even though each node currenly only has 2 centroid)
    }
    BeliefExporter be(dn, 0);

    int len = 41;
    int b0 = 0;
    int b1 = ( 32768 / 3 ) * 1;
    int b2 = ( 32768 / 3 ) * 2;
    assertIntEquals(len, be.getWinningCentroidTreeSize());

    printf("b1: %i, b2: %i\n", b1, b2);
    short * t = be.getWinningCentroidTree();
    for(int i = 0 ; i < 41; i++){
        printf("%i ",t[i]);
    }
    printf("\n");
    //create a tree from the winning centroids
    assertShortArrayEqualsV(be.getWinningCentroidTree(), len,
        b2 + 20,
        b1 + 16,  0, -1,  1, -1,  4, -1,  5, -1, -1,
        b1 + 17,  2, -1,  3, -1,  6, -1,  7, -1, -1,
        b1 + 18,  8, -1,  9, -1, 12, -1, 13, -1, -1,
        b1 + 19, 10, -1, 11, -1, 14, -1, 15, -1, -1)

    return 0;
}

int experiment(){
    uint centroids[8] = {2,4,4,16,8,8,4,4};
    DestinNetworkAlt dn(W512, 8, centroids, true);
    dn.setBeliefTransform(DST_BT_P_NORM);
    float temperatures[8] = {2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0};
    dn.setTemperatures(temperatures);
    dn.setIsPOSTraining(true);
    VideoSource vs(false, "circle_square.mp4");

    for(int i = 800 ; i--; ){
        if(!vs.grab()){
            continue;
        }

        dn.doDestin(vs.getOutput());
    }
    dn.setIsPOSTraining(false);

    BeliefExporter be(dn, 0);
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
        tmw.addTreeToDatabase(be.getWinningCentroidTree(), len);
    }
    vector<PatternTree> minedTrees;
    int support = 10;
    tmw.mine(support, minedTrees);
    vector<short> atree;
    tmw.treeToVector(minedTrees.at(0), atree);
    dn.displayTree(atree);

}

int main(int argc, char ** argv){
    RUN(testTreeToVector);
    RUN(testTreeMiner);
    //RUN(testBeliefExporter);
    UT_REPORT_RESULTS();
    return 0;
}

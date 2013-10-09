#include "unit_test.h"
#include "DestinNetworkAlt.h"
#include <ostream>

using namespace std;

int regressionTest();

int main(int argc, char ** argv){
    RUN(regressionTest);
    return 0;
}

int regressionTest(){
    dst_ut_srand(12345); // seed our test random number generator
    srand(101); // seed the normal random number generator

    uint centroids[4] = {4,4,4,4};
    DestinNetworkAlt dna(W32, 4,  centroids, true);

    int nImages = 10;
    float ** images = makeRandomImages(32*32, nImages);
    int iterations = 50;
    SetLearningStrat(dna.getNetwork(), CLS_FIXED);
    dna.setFixedLearnRate(0.1);

    for(int i = 0 ; i < iterations ; i++ ){
        dna.doDestin(images[i % nImages]);
    }

    cout << dna.getNode(3,0,0)->beliefMal[0] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[1] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[2] << endl;
    cout << dna.getNode(3,0,0)->beliefMal[3] << endl;

    freeRandomImages(images, nImages);
}

int testAddCentroid(){
    uint centroids[4] = {4,4,4,4};
    DestinNetworkAlt dna(W32, 4,  centroids, true);

}

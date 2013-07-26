
#include <stdlib.h>
#include <vector>
#include "SomPresentor.h"
#include "ClusterSom.h"

void test_org();
void test_SOM_demo();

int main(int argc, char ** argv){
    //test_org();
    test_SOM_demo();

    return 0;
}

void test_org()
{
    ClusterSom som(50,50,3);

    std::vector<std::vector<float> > data;

    int n = 1000;
    for(int i = 0 ; i < n; i++){

        std::vector<float> d;
        for(int j = 0 ; j < 3 ; j++){
            d.push_back( (float)rand() / RAND_MAX );
        }
        data.push_back(d);
        som.addTrainData(d.data());
    }

    SomPresentor sp(som);

    som.train(5000);

    sp.showSimularityMap();

    // pause for inut to let user see the output window
    cv::waitKey();
}

void test_SOM_demo()
{
    // 2013.7.19
    // This is a simple demo to show how to use SOM to train high-dim vectors,
    // then draw those vectors on a 2-dim space with different colors.

    ClusterSom som(50,50,3);

    std::vector<std::vector<float> > data;

    int n = 1000;
    for(int i = 0 ; i < n; i++){

        std::vector<float> d;
        for(int j = 0 ; j < 3 ; j++){
            d.push_back( (float)rand() / RAND_MAX );
        }
        data.push_back(d);
        som.addTrainData(d.data());
    }

    SomPresentor sp(som);

    som.train(5000);

    // CZT: split the inputs into 2 parts, then draw then with different color;
    for(int i=0; i<n/2; ++i)
    {
        std::vector<float> d = data.at(i);
        CvPoint cp = som.findBestMatchingUnit(d.data());
        sp.addSimMapMaker(cp.y, cp.x, 0.1, 5);
    }
    for(int i=n/2; i<n; ++i)
    {
        std::vector<float> d = data.at(i);
        CvPoint cp = som.findBestMatchingUnit(d.data());
        sp.addSimMapMaker(cp.y, cp.x, 0.5, 5);
    }

    sp.showSimularityMap();

    // pause for inut to let user see the output window
    cv::waitKey();
}

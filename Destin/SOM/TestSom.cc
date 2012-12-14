
#include <stdlib.h>
#include "BrlySom.hpp"
#include "SomPresentor.hpp"


int main(int argc, char ** argv){
    BrlySom som(50, 50, 3);


    int n = 1000;
    for(int i = 0 ; i < n; i++){
        float data[3];
        for(int j = 0 ; j < 3 ; j++){
            data[j] = (float)rand() / RAND_MAX;
        }
        som.train_iterate(data);
    }

    SomPresentor sp(som);

    sp.showSimularityMap();

    // pause for inut to let user see the output window
    cv::waitKey();

}


#include "Som.hpp"
#include <stdlib.h>


int main(int argc, char ** argv){
    Som som(50, 50, 3);

    int n = 1000;
    for(int i = 0 ; i < n; i++){
        float data[3];
        for(int j = 0 ; j < 3 ; j++){
            data[j] = (float)rand() / RAND_MAX;
        }
        som.train_iterate(data);
    }
    som.showSimularityMap("Simularity", 200,200);

}

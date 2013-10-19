#ifndef __CENTROID_H
#define __CENTROID_H

#include "destin.h"

void InitUniformCentroids(
    Destin *destin,   // initialized destin pointer
    uint l,           // layer
    uint nb,          // number of centroids
    uint ns           // number of states (centroid dimensionality)
);


#endif
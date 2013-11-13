#ifndef __CENTROID_H
#define __CENTROID_H

#include "destin.h"

void InitUniformCentroids(
        Destin *destin,   // initialized destin pointer
        uint l,           // layer
        uint ni,          // input dimensionality
        uint nb,          // number of centroids
        uint np,          // number of parent centroids
        uint ns           // number of states (centroid dimensionality)
     );

void DeleteUniformCentroid(
        Destin *destin,   // initialized destin pointer
        uint l,           // layer
        uint idx          // index of deleted centroid
     );

void AddUniformCentroid(
        Destin *destin,   // initialized destin pointer
        uint l            // layer
     );

#endif

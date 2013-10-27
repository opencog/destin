#ifndef __CENTROID_H
#define __CENTROID_H

#include "destin.h"

void InitUniformCentroids(
        Destin *destin,   // initialized destin pointer
        uint l,           // layer
        uint nb,          // number of centroids
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

// used to initialize mu/sigma for added centroids (one of methods)
// new values are calculated by averaging nearest neighbour centroids values
void InitUniformCentroidByAvgNearNeighbours(
        Destin *destin,   // initialized destin potiner
        uint l,           // layer
        uint idx,         // index of updated centroid
        uint nearest      // number of nearest centroids used for averaging
     );

#endif
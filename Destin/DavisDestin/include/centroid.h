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

// The method used to distribute values of beliefs associated with centroid that
// is going to be deleted. It helps when deleted centroid is very close to good
// learned centroids. This centroids from neighbourhood get additional weighted
// belief of deleted centroid where more weights have centroids that are closer.
void DistributeEvidenceOfDeletedCentroidToNeighbours(
        Destin *destin,   // initialized destin potiner
        uint l,           // layer
        uint idx,         // index of updated centroid
        uint nearest      // number of nearest centroids used for averaging
     );

#endif
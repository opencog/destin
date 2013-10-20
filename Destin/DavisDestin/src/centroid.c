#include <stdlib.h>

#include "macros.h"
#include "centroid.h"
#include "destin.h"
#include "array.h"

void InitUniformCentroids(Destin *d, uint l, uint nb, uint ns)
{
    uint i, j;

    MALLOCV(d->uf_mu[l], float *, nb);
    MALLOCV(d->uf_sigma[l], float *, nb);
    MALLOCV(d->uf_absvar[l], float, ns);

    MALLOCV(d->uf_winCounts[l], uint, nb);
    MALLOCV(d->uf_persistWinCounts[l], long, nb);
    MALLOCV(d->uf_persistWinCounts_detailed[l], long, nb);
    MALLOCV(d->uf_starv[l], float, nb);

    MALLOCV(d->uf_avgDelta[l], float *, nb);
    MALLOCV(d->uf_avgSquaredDelta[l], float *, nb);
    MALLOCV(d->uf_avgAbsDelta[l], float, ns);

    for (i=0; i < nb; i++)
    {
        MALLOCV(d->uf_mu[l][i], float, ns);
        MALLOCV(d->uf_sigma[l][i], float, ns);
        MALLOCV(d->uf_avgDelta[l][i], float, ns);
        MALLOCV(d->uf_avgSquaredDelta[l][i], float, ns);

        d->uf_winCounts[l][i] = 0;
        d->uf_persistWinCounts[l][i] = 0;
        d->uf_persistWinCounts_detailed[l][i] = 0;
        d->uf_starv[l][i] = 1;

        for (j=0; j < ns; j++)
        {
            d->uf_mu[l][i][j] = (float) rand() / (float) RAND_MAX;
            d->uf_sigma[l][i][j] = INIT_SIGMA;
        }
    }

    for (j=0; j < ns; j++)
    {
        d->uf_absvar[l][j] = 0;
    }
}

void DeleteUniformCentroid(Destin *d, uint l, uint idx)
{
    uint i, j, ni, nb, ns;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ni = n->ni;
    ns = n->ns;

    // Layer l
    for (i = 0; i < nb; i++)
    {
        ArrayDeleteFloat(&d->uf_mu[l][i], ns, ni+idx);
        ArrayDeleteFloat(&d->uf_sigma[l][i], ns, ni+idx);
        ArrayDeleteFloat(&d->uf_avgDelta[l][i], ns, ni+idx);
        ArrayDeleteFloat(&d->uf_avgSquaredDelta[l][i], ns, ni+idx);
    }
    ArrayDeleteFloat(&d->uf_absvar[l], ns, ni+idx);
    ArrayDeleteFloat(&d->uf_avgAbsDelta[l], ns, ni+idx);

    ArrayDeleteArray((void *)&d->uf_mu[l], nb, idx);
    ArrayDeleteArray((void *)&d->uf_sigma[l], nb, idx);
    ArrayDeleteUInt(&d->uf_winCounts[l], nb, idx);
    ArrayDeleteLong(&d->uf_persistWinCounts[l], nb, idx);
    ArrayDeleteLong(&d->uf_persistWinCounts_detailed[l], nb, idx);
    ArrayDeleteArray((void *)&d->uf_avgDelta[l], nb, idx);
    ArrayDeleteArray((void *)&d->uf_avgSquaredDelta[l], nb, idx);
    ArrayDeleteFloat(&d->uf_avgAbsDelta[l], nb, idx);
    ArrayDeleteFloat(&d->uf_starv[l], nb, idx);

    d->nb[l]--;  // decrease global number of centroids for layer l
    for (j = 0; j < d->layerSize[l]; j++)
    {
        n =& d->nodes[d->layerNodeOffsets[l] + j];

        ArrayDeleteFloat(&n->belief, nb, idx);
        ArrayDeleteFloat(&n->beliefEuc, nb, idx);
        ArrayDeleteFloat(&n->beliefMal, nb, idx);
        ArrayDeleteFloat(&n->outputBelief, nb, idx);
        ArrayDeleteFloat(&n->delta, ns, ni+idx);
        ArrayDeleteFloat(&n->observation, ns, ni+idx);

        // decrease centroid dimensionality for each node from layer l
        UpdateNodeSizes(n, n->ni, nb-1, n->np, n->nc);

        // pointers may change due to reallocation
        n->mu = d->uf_mu[l];
        n->starv = d->uf_starv[l];
    }

    // Layer l+1
    if (l+1 < d->nLayers)
    {
        n = GetNodeFromDestin(d, l+1, 0, 0);
        ns = n->ns;
        uint childIndexes[n->childNumber];  // indexes of deleted centroids for all childs

        for (i = 0; i < n->childNumber; i++)
        {
            childIndexes[i] = i*nb + idx;
        }
        for (i = 0; i < n->nb; i++)
        {
            ArrayDeleteFloats(&d->uf_mu[l+1][i], ns, childIndexes, n->childNumber);
            ArrayDeleteFloats(&d->uf_sigma[l+1][i], ns, childIndexes, n->childNumber);
            ArrayDeleteFloats(&d->uf_avgDelta[l+1][i], ns, childIndexes, n->childNumber);
            ArrayDeleteFloats(&d->uf_avgSquaredDelta[l+1][i], ns, childIndexes, n->childNumber);
        }
        ArrayDeleteFloats(&d->uf_absvar[l], ns, childIndexes, n->childNumber);
        ArrayDeleteFloats(&d->uf_avgAbsDelta[l], ns, childIndexes, n->childNumber);

        for (j = 0; j < d->layerSize[l+1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l+1] + j];

            ArrayDeleteFloats(&n->delta, ns, childIndexes, n->childNumber);
            ArrayDeleteFloats(&n->observation, ns, childIndexes, n->childNumber);

            // decrease centroid dimensionality for each node from layer l+1
            UpdateNodeSizes(n, n->ni - n->childNumber, n->nb, n->np, n->nc);
        }
    }

    // Layer l-1
    if (l > 0)
    {
        n = GetNodeFromDestin(d, l-1, 0, 0);
        ns = n->ns;
        uint pIdx = n->ni + n->nb + idx; // index of deleted parent centroid

        for (i = 0; i < n->nb; i++)
        {
            ArrayDeleteFloat(&d->uf_mu[l-1][i], ns, pIdx);
            ArrayDeleteFloat(&d->uf_sigma[l-1][i], ns, pIdx);
            ArrayDeleteFloat(&d->uf_avgDelta[l-1][i], ns, pIdx);
            ArrayDeleteFloat(&d->uf_avgSquaredDelta[l-1][i], ns, pIdx);
        }
        ArrayDeleteFloat(&d->uf_absvar[l-1], ns, pIdx);
        ArrayDeleteFloat(&d->uf_avgAbsDelta[l-1], ns, pIdx);

        for (j = 0; j < d->layerSize[l-1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l-1] + j];

            ArrayDeleteFloat(&n->delta, ns, pIdx);
            ArrayDeleteFloat(&n->observation, ns, pIdx);

            // decrease centroid dimensionality for each node from layer l-1
            UpdateNodeSizes(n, n->ni, n->nb, n->np-1, n->nc);
        }
    }

}

void AddUniformCentroid(Destin *d, uint l)
{
    uint i, j, ni, nb, ns, idx;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ni = n->ni;
    ns = n->ns;
    idx = ni + nb;

    // Layer l
    float * newMu, * newSigma, * newAvgDelta, * newAvgSquaredDelta;

    MALLOCV(newMu, float, ns+1);
    MALLOCV(newSigma, float, ns+1);
    MALLOCV(newAvgDelta, float, ns+1);
    MALLOCV(newAvgSquaredDelta, float, ns+1);

    // TODO: initialization depends on method
    // TODO: 1/ns, 1/ni etc.
    for (j=0; j < ns; j++)
    {
        d->uf_mu[l][i][j] = (float) rand() / (float) RAND_MAX;
        d->uf_sigma[l][i][j] = INIT_SIGMA;
    }
    for (i = 0; i < nb; i++)
    {
        ArrayInsertFloat(&d->uf_mu[l][i], ns, idx, 0);
        ArrayInsertFloat(&d->uf_sigma[l][i], ns, idx, INIT_SIGMA);
        ArrayInsertFloat(&d->uf_avgDelta[l][i], ns, idx, 0);
        ArrayInsertFloat(&d->uf_avgSquaredDelta[l][i], ns, idx, 0);
    }
    ArrayInsertFloat(&d->uf_absvar[l], ns, idx, 0);
    ArrayInsertFloat(&d->uf_avgAbsDelta[l], ns, idx, 0);

    ArrayAppendPtr((void *)&d->uf_mu[l], nb, newMu);
    ArrayAppendPtr((void *)&d->uf_sigma[l], nb, newSigma);
    ArrayAppendUInt(&d->uf_winCounts[l], nb, 0);
    ArrayAppendLong(&d->uf_persistWinCounts[l], nb, 0);
    ArrayAppendLong(&d->uf_persistWinCounts_detailed[l], nb, 0);
    ArrayAppendPtr((void *)&d->uf_avgDelta[l], nb, newAvgDelta);
    ArrayAppendPtr((void *)&d->uf_avgSquaredDelta[l], nb, newAvgSquaredDelta);
    ArrayAppendFloat(&d->uf_avgAbsDelta[l], nb, 0);
    ArrayAppendFloat(&d->uf_starv[l], nb, 1);

    d->nb[l]++;  // increase global number of centroids for layer l
    for (j = 0; j < d->layerSize[l]; j++)
    {
        n =& d->nodes[d->layerNodeOffsets[l] + j];

        ArrayAppendFloat(&n->belief, nb, 0);
        ArrayAppendFloat(&n->beliefEuc, nb, 0);
        ArrayAppendFloat(&n->beliefMal, nb, 0);
        ArrayAppendFloat(&n->outputBelief, nb, 0);
        ArrayInsertFloat(&n->delta, ns, idx, 0);
        ArrayInsertFloat(&n->observation, ns, idx, 0);

        // increase centroid dimensionality for each node from layer l
        UpdateNodeSizes(n, n->ni, nb+1, n->np, n->nc);

        // pointers may change due to reallocation
        n->mu = d->uf_mu[l];
        n->starv = d->uf_starv[l];
    }

    // Layer l+1
    if (l+1 < d->nLayers)
    {
        n = GetNodeFromDestin(d, l+1, 0, 0);
        ns = n->ns;
        uint childIndexes[n->childNumber];  // indexes of added centroids for all childs
        float childValues[n->childNumber];  // initial values for all childs
        float childSigmas[n->childNumber];

        for (i = 0; i < n->childNumber; i++)
        {
            childIndexes[i] = i*nb;
            childValues[i] = 0;
            childSigmas[i] = INIT_SIGMA;
        }
        for (i = 0; i < n->nb; i++)
        {
            ArrayInsertFloats(&d->uf_mu[l+1][i], ns, childIndexes, childValues, n->childNumber);
            ArrayInsertFloats(&d->uf_sigma[l+1][i], ns, childIndexes, childSigmas, n->childNumber);
            ArrayInsertFloats(&d->uf_avgDelta[l+1][i], ns, childIndexes, childValues, n->childNumber);
            ArrayInsertFloats(&d->uf_avgSquaredDelta[l+1][i], ns, childIndexes, childValues, n->childNumber);
        }
        ArrayInsertFloats(&d->uf_absvar[l], ns, childIndexes, childValues, n->childNumber);
        ArrayInsertFloats(&d->uf_avgAbsDelta[l], ns, childIndexes, childValues, n->childNumber);

        for (j = 0; j < d->layerSize[l+1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l+1] + j];

            ArrayInsertFloats(&n->delta, ns, childIndexes, childValues, n->childNumber);
            ArrayInsertFloats(&n->observation, ns, childIndexes, childValues, n->childNumber);

            // increase centroid dimensionality for each node from layer l+1
            UpdateNodeSizes(n, n->ni + n->childNumber, n->nb, n->np, n->nc);
        }
    }

    // Layer l-1
    if (l > 0)
    {
        n = GetNodeFromDestin(d, l-1, 0, 0);
        ns = n->ns;
        uint pIdx = n->ni + n->nb + n->np;   // index of added parent centroid

        for (i = 0; i < n->nb; i++)
        {
            ArrayInsertFloat(&d->uf_mu[l-1][i], ns, pIdx, 0);
            ArrayInsertFloat(&d->uf_sigma[l-1][i], ns, pIdx, INIT_SIGMA);
            ArrayInsertFloat(&d->uf_avgDelta[l-1][i], ns, pIdx, 0);
            ArrayInsertFloat(&d->uf_avgSquaredDelta[l-1][i], ns, pIdx, 0);
        }
        ArrayInsertFloat(&d->uf_absvar[l-1], ns, pIdx, 0);
        ArrayInsertFloat(&d->uf_avgAbsDelta[l-1], ns, pIdx, 0);

        for (j = 0; j < d->layerSize[l-1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l-1] + j];

            ArrayInsertFloat(&n->delta, ns, pIdx, 0);
            ArrayInsertFloat(&n->observation, ns, pIdx, 0);

            // increase centroid dimensionality for each node from layer l-1
            UpdateNodeSizes(n, n->ni, n->nb, n->np+1, n->nc);
        }
    }
}


// Private structure for q-sorting neighbouring centroids
// Neighbours are sorted by distance from given centroid
typedef struct _Neighbour {
    uint index;      // index in centroid arrays (0..nb-1)
    float distance;  // distance from given centroid
} _Neighbour;

// Private comparator of neighbouring centroids
int _compareNeighbours(void * n1, void * n2)
{
    float d1 = ((_Neighbour *) n1)->distance;
    float d2 = ((_Neighbour *) n2)->distance;
    return (d1 > d2) - (d2 > d1);
}

float _calcNeighboursDistanceEuc(float * mu1, float * mu2, uint ns)
{
    uint i;
    float delta, dist = 0;

    for (i = 0; i < ns; i++)
    {
        delta = mu1[i] - mu2[i];
        dist += delta * delta;
    }
    return dist;
}

#define UpdateAvgNearNeighbours(array, neighbours, count, offset, idx) {\
    uint c;                                          \
    float sum;                                       \
    for (c = 1; c <= nearest; c++) {                 \
        sum += array[offset + neighbours[c].index];  \
    }                                                \
    array[offset + idx] = sum/nearest;               \
}

void InitUniformCentroidByAvgNearNeighbours(Destin *d, uint l, uint idx, uint nearest)
{
    uint i, j, ni, nb, ns, count;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    ni = n->ni;
    nb = n->nb;
    ns = n->ns;

    _Neighbour neighbours[nb];
    for (i = 0; i < nb; i++)
    {
        neighbours[i].index = i;
        neighbours[i].distance = _calcNeighboursDistanceEuc(d->uf_mu[l][i], d->uf_mu[l][idx], ns);
    }
    qsort((void *)neighbours, nb, sizeof(_Neighbour), (void *)&_compareNeighbours);
    nearest = _MIN(nearest, nb-1);

    // Layer l
    for (i = 0; i < nb; i++)
    {
        UpdateAvgNearNeighbours(d->uf_mu[l][i], neighbours, nearest, ni, idx);
        UpdateAvgNearNeighbours(d->uf_sigma[l][i], neighbours, nearest, ni, idx);
    }

    // Layer l+1
    if (l+1 < d->nLayers)
    {
        n = GetNodeFromDestin(d, l+1, 0, 0);
        for (i = 0; i < n->nb; i++)
        {
            for (j = 0; j < n->childNumber; j++)
            {
                UpdateAvgNearNeighbours(d->uf_mu[l+1][i], neighbours, nearest, j*nb, idx);
                UpdateAvgNearNeighbours(d->uf_sigma[l+1][i], neighbours, nearest, j*nb, idx);
            }
        }
    }

    // Layer l-1
    if (l > 0)
    {
        n = GetNodeFromDestin(d, l-1, 0, 0);
        uint offset = n->ni + n->nb;

        for (i = 0; i < n->nb; i++)
        {
            UpdateAvgNearNeighbours(d->uf_mu[l-1][i], neighbours, nearest, offset, idx);
            UpdateAvgNearNeighbours(d->uf_sigma[l-1][i], neighbours, nearest, offset, idx);
        }
    }
}


/*****************************************************************************/
/*
// 2013.5.31, 2013.7.4
// addCentroid
// Keep 'shareCentroids', 'uf_starv', 'uf_sigma', 'uf_avgDelta',
// 'uf_persistWinCounts', 'uf_persistWinCounts_detailed', 'uf_absvar';
void addCentroid(Destin * d, uint *nci, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma, float *temp, float starvCoeff, uint nMovements, bool isUniform,
                  int extRatio, int currLayer, float ** sharedCen, float ** starv, float ** sigma,
                  long **persistWinCounts, long ** persistWinCounts_detailed, float ** absvar)
{
    uint i, l, maxNb, maxNs, j;

    initializeDestinParameters(nb, isUniform, nci, extRatio, nl, nMovements, d, nc, temp);

    if(isUniform){
        // uf_starv
        // uf_persistWinCounts
        for(l=0; l<d->nLayers; ++l)
        {
            int tempEnd = d->nb[l];
            if(l == currLayer)
            {
                tempEnd--;
                d->uf_starv[l][tempEnd] = 1;
                d->uf_persistWinCounts[l][tempEnd] = 0;
                d->uf_persistWinCounts_detailed[l][tempEnd] = 0;
            }
            for(i=0; i<tempEnd; ++i)
            {
                d->uf_starv[l][i] = starv[l][i];
                d->uf_persistWinCounts[l][i] = persistWinCounts[l][i];
                d->uf_persistWinCounts_detailed[l][i] = persistWinCounts_detailed[l][i];
            }
        }
    }

    // keep track of the max num of beliefs and states.  we need this information
    // to correctly call kernels later
    maxNb = 0;
    maxNs = 0;

    uint n = 0;

    // New method
    // My thought: initialize the new centroid and get the related information
    // just before running!
    int niCurr = currLayer==0 ? d->nci[0] : d->nci[currLayer] * d->nb[currLayer-1];
    int npCurr = currLayer==nl-1 ? 0 : d->nb[currLayer+1];
    int nsCurr = currLayer==0 ? niCurr*extRatio+d->nb[currLayer]+npCurr : niCurr+d->nb[currLayer]+npCurr;
    float * newCentroid;
    float * cenDis;
    int * cenIndex;
    MALLOC(newCentroid, float, (nsCurr-1));  // ()!!!
    MALLOC(cenDis, float, (d->nb[currLayer]-1)); // ()!!!
    MALLOC(cenIndex, int, (d->nb[currLayer]-1)); // ()!!!
    for(i=0; i<nsCurr-1; ++i)
    {
        newCentroid[i] = (float)rand() / (float)RAND_MAX;
    }
    for(i=0; i<d->nb[currLayer]-1; ++i)
    {
        cenDis[i] = 0.0;
        cenIndex[i] = i;
    }
    for(i=0; i<d->nb[currLayer]-1; ++i)
    {
        for(j=0; j<nsCurr-1; ++j)
        {
            float fTemp = sharedCen[currLayer][i*(nsCurr-1)+j] - newCentroid[j];
            fTemp *= fTemp;
            cenDis[i] += fTemp;
        }
    }

    for(i=0; i<d->nb[currLayer]-1-1; ++i)
    {
        for(j=i+1; j<d->nb[currLayer]-1; ++j)
        {
            if(cenDis[i] > cenDis[j])
            {
                float fTemp = cenDis[i];
                cenDis[i] = cenDis[j];
                cenDis[j] = fTemp;

                int iTemp = cenIndex[i];
                cenIndex[i] = cenIndex[j];
                cenIndex[j] = iTemp;
            }
        }
    }
    int iCount = d->nb[currLayer]-1>5 ? 5 : d->nb[currLayer]-1;
    MALLOC(d->nearInd, int, iCount);
    for(i=0; i<iCount; ++i)
    {
        d->nearInd[i] = cenIndex[i];
    }
    d->sizeInd = iCount;


    // ---END OF NEW METHOD---

    // initialize the rest of the network

    for( l=0; l < nl; l++ )
    {
        // update max belief
        if( nb[l] > maxNb )
        {
            maxNb = nb[l];
        }

        float * sharedCentroids;

        uint np = ((l + 1 == nl) ? 0 : nb[l+1]);
        uint ni = (l == 0 ? d->nci[0] : d->nci[l] * nb[l-1]);
        uint ns = nb[l] + np + nc + ((l == 0) ? ni*extRatio : ni);

        if (ns > maxNs)
        {
            maxNs = ns;
        }

        if(isUniform){
            MALLOC(d->uf_avgDelta[l], float, ns*nb[l]);
            MALLOC(sharedCentroids, float, ns*nb[l]);
            MALLOC(d->uf_sigma[l], float, ns*nb[l]);

            // 2013.7.4
            // CZT: uf_absvar;
            MALLOC(d->uf_absvar[l], float, ns*nb[l]);

            //
            MALLOC(d->uf_avgSquaredDelta[l], float, ns*nb[l]);
            MALLOC(d->uf_avgAbsDelta[l], float, ns*nb[l]);

            // 2013.6.4
            // Keep the shareCentroids, uf_sigma
//#define USE_METHOD1
            if(l == currLayer)
            {
                int tempI, tempJ;
                int tempRange = l==0?ni*(d->extRatio-1):0;
                for(tempI=0; tempI<d->nb[l]-1; ++tempI)
                {
                    for(tempJ=0; tempJ<ns-np-1-tempRange; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns-1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns-1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns-1)+tempJ];
                    }
#ifdef USE_METHOD1
                    // New method
                    float fSum = 0.0;
                    for(tempJ=0; tempJ<d->sizeInd; ++tempJ)
                    {
                        fSum += sharedCen[l][tempI*(ns-1)+ni+d->nearInd[tempJ]];
                    }
                    sharedCentroids[tempI*ns+ns-np-1-tempRange] = fSum/(float)d->sizeInd;
                    fSum = 0.0;
                    for(tempJ=0; tempJ<d->sizeInd; ++tempJ)
                    {
                        fSum += sigma[l][tempI*(ns-1)+ni+d->nearInd[tempJ]];
                    }
                    d->uf_sigma[l][tempI*ns+ns-np-1-tempRange] = fSum/(float)d->sizeInd;
#endif
#ifndef USE_METHOD1
                    sharedCentroids[tempI*ns+ns-np-1-tempRange] = 0.0;
                    d->uf_sigma[l][tempI*ns+ns-np-1-tempRange] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][tempI*ns+ns-np-1-tempRange] = 0.0;
#endif
                    for(tempJ=ns-np-tempRange; tempJ<ns; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns-1)+tempJ-1];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns-1)+tempJ-1];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns-1)+tempJ-1];
                    }
                }
#ifdef USE_METHOD1
                for(i=ns*(d->nb[l]-1); i<ns*d->nb[l]-np-tempRange-1; ++i)
                {
                    sharedCentroids[i] = newCentroid[i-ns*(d->nb[l]-1)];
                    d->uf_sigma[l][i] = INIT_SIGMA;
                }
                float fSum = 0.0;
                for(tempI=0; tempI<d->sizeInd; ++tempI)
                {
                    fSum += newCentroid[ni+d->nearInd[tempI]];
                }
                sharedCentroids[ns*d->nb[l]-np-tempRange-1] = fSum/(float)d->sizeInd;
                d->uf_sigma[l][ns*d->nb[l]-np-tempRange-1] = INIT_SIGMA;
                for(i=ns*d->nb[l]-np-tempRange; i<ns*d->nb[l]; ++i)
                {
                    sharedCentroids[i] = newCentroid[i-ns*(d->nb[l]-1)-1];
                    d->uf_sigma[l][i] = INIT_SIGMA;
                }
#endif
#ifndef USE_METHOD1
                for(i=ns*(d->nb[l]-1); i<ns*d->nb[l]-nb[l]-np-tempRange; ++i)
                {
                    sharedCentroids[i] = 1/(float)(ns-nb[l]-np-tempRange);
                    d->uf_sigma[l][i] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][i] = 0.0;
                }
                for(i=ns*d->nb[l]-nb[l]-np-tempRange; i<ns*d->nb[l]-np-tempRange; ++i)
                {
                    sharedCentroids[i] = 1/(float)(nb[l]);
                    d->uf_sigma[l][i] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][i] = 0.0;
                }
                for(i=ns*d->nb[l]-np-tempRange; i<ns*d->nb[l]-tempRange; ++i)
                {
                    sharedCentroids[i] = 1/(float)(np);
                    d->uf_sigma[l][i] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][i] = 0.0;
                }
                for(i=ns*d->nb[l]-tempRange; i<ns*d->nb[l]; ++i)
                {
                    sharedCentroids[i] = 1/(float)(tempRange);
                    d->uf_sigma[l][i] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][i] = 0.0;
                }
#endif
            }
            // currLayer-1
            else if(l==currLayer-1)
            {
                int tempI, tempJ;
                int tempRange = l==0?ni*(d->extRatio-1):0;
                for(tempI=0; tempI<d->nb[l]; ++tempI)
                {
                    for(tempJ=0; tempJ<ns-1-tempRange; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns-1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns-1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns-1)+tempJ];
                    }
#ifdef USE_METHOD1
                    // New method
                    float fSum = 0.0;
                    for(tempJ=0; tempJ<d->sizeInd; ++tempJ)
                    {
                        fSum += sharedCen[l][tempI*(ns-1)+ni+d->nb[l]+d->nearInd[tempJ]];
                    }
                    sharedCentroids[tempI*ns+ns-1-tempRange] = fSum/(float)d->sizeInd;
                    fSum = 0.0;
                    for(tempJ=0; tempJ<d->sizeInd; ++tempJ)
                    {
                        fSum += sigma[l][tempI*(ns-1)+ni+d->nb[l]+d->nearInd[tempJ]];
                    }
                    d->uf_sigma[l][tempI*ns+ns-1-tempRange] = fSum/(float)d->sizeInd;
#endif
#ifndef USE_METHOD1
                    sharedCentroids[tempI*ns+ns-1-tempRange] = 0.0;
                    d->uf_sigma[l][tempI*ns+ns-1-tempRange] = INIT_SIGMA;
                    //
                    d->uf_absvar[l][tempI*ns+ns-1-tempRange] = 0.0;
#endif
                    for(tempJ=ns-tempRange; tempJ<ns; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns-1)+tempJ-1];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns-1)+tempJ-1];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns-1)+tempJ-1];
                    }
                }
            }
            // currLayer+1
            else if(l==currLayer+1)
            {
                int tempI, tempJ, tempK;
                uint childs = d->nci[l];
                for(tempI=0; tempI<d->nb[l]; ++tempI)
                {
                    for(tempJ=0; tempJ<childs; ++tempJ)
                    {
                        for(tempK=0; tempK<d->nb[l-1]-1; ++tempK)
                        {
                            sharedCentroids[tempI*ns+tempJ*d->nb[l-1]+tempK] = sharedCen[l][tempI*(ns-childs)+tempJ*(d->nb[l-1]-1)+tempK];
                            d->uf_sigma[l][tempI*ns+tempJ*d->nb[l-1]+tempK] = sigma[l][tempI*(ns-childs)+tempJ*(d->nb[l-1]-1)+tempK];
                            //
                            d->uf_absvar[l][tempI*ns+tempJ*d->nb[l-1]+tempK] = absvar[l][tempI*(ns-childs)+tempJ*(d->nb[l-1]-1)+tempK];
                        }
#ifdef USE_METHOD1
                        // New method
                        float fSum = 0.0;
                        for(tempK=0; tempK<d->sizeInd; ++tempK)
                        {
                            fSum += sharedCen[l][tempI*(ns-childs)+tempJ*(d->nb[l-1]-1)+d->nearInd[tempK]];
                        }
                        sharedCentroids[tempI*ns+tempJ*d->nb[l-1]+d->nb[l-1]-1] = fSum/(float)d->sizeInd;
                        fSum = 0.0;
                        for(tempK=0; tempK<d->sizeInd; ++tempK)
                        {
                            fSum += sigma[l][tempI*(ns-childs)+tempJ*(d->nb[l-1]-1)+d->nearInd[tempK]];
                        }
                        d->uf_sigma[l][tempI*ns+tempJ*d->nb[l-1]+d->nb[l-1]-1] = fSum/(float)d->sizeInd;
#endif
#ifndef USE_METHOD1
                        sharedCentroids[tempI*ns+tempJ*d->nb[l-1]+d->nb[l-1]-1] = 0.0;
                        d->uf_sigma[l][tempI*ns+tempJ*d->nb[l-1]+d->nb[l-1]-1] = INIT_SIGMA;
                        //
                        d->uf_absvar[l][tempI*ns+tempJ*d->nb[l-1]+d->nb[l-1]-1] = 0.0;
#endif
                    }
                    for(tempJ=childs*d->nb[l-1]; tempJ<ns; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns-childs)+tempJ-childs];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns-childs)+tempJ-childs];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns-childs)+tempJ-childs];
                    }
                }
            }
            else
            {
                for(i=0; i<ns*d->nb[l]; ++i)
                {
                    sharedCentroids[i] = sharedCen[l][i];
                    d->uf_sigma[l][i] = sigma[l][i];
                    //
                    d->uf_absvar[l][i] = absvar[l][i];
                }
            }
        }else{
            sharedCentroids = NULL;
        }


        uint inputOffsets[ni];
        for( i=0; i < d->layerSize[l]; i++, n++ )
        {
            if (l == 0)
            {
                CalcSquareNodeInputOffsets(d->layerWidth[0], i, ni, inputOffsets);
            }

            InitNode(
                        n,
                        d,
                        l,
                        ni,
                        nb[l],
                        np,
                        nc,
                        ns,
                        starvCoeff,
                        beta,
                        gamma,
                        lambda,
                        temp[l],
                        &d->nodes[n],
                        (l > 0 ? NULL : inputOffsets),
                        (l > 0 ? d->nci[l] : 0)
                    );
        }//next node
    }//next layer

    LinkParentsToChildren( d );
    d->maxNb = maxNb;
    d->maxNs = maxNs;

    // 2013.7.3
    // CZT: should FREE to avoid memory leak;
    for(i=0; i<nl; ++i)
    {
        FREE(sharedCen[i]);
        FREE(starv[i]);
        FREE(sigma[i]);
        FREE(persistWinCounts[i]);
        FREE(persistWinCounts_detailed[i]);
        FREE(absvar[i]);
    }
    FREE(sharedCen);
    FREE(starv);
    FREE(sigma);
    FREE(persistWinCounts);
    FREE(persistWinCounts_detailed);
    FREE(absvar);
}

*/

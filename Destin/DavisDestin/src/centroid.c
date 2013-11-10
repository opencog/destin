#include <stdlib.h>

#include "macros.h"
#include "centroid.h"
#include "destin.h"
#include "array.h"

void _normalizeFloatArray(float * array, uint length)
{
    uint i;
    float sum = 0;

    for (i = 0; i < length; i++)
    {
        sum += array[i];
    }
    for (i = 0; i < length; i++)
    {
        array[i] = (sum > 0) ? array[i] / sum : 1 / (float) length;
    }
}

void _normalizeMu(float * mu, uint ni, uint nb, uint np)
{
    uint i;

    // TODO: in recursive mode previous belief and parent belief should be normalized to 1

    // Recursive mode off
    for (i = ni; i < ni+nb; i++)
    {
        mu[i] = 1/(float) nb;
    }
    // Recursive mode off
    for (i = ni+nb; i < ni+nb+np; i++)
    {
        mu[i] = 1/(float) np;
    }
}

void _initUniformCentroidMu(float * mu, uint ni, uint nb, uint np, uint ns)
{
    uint i;

    for (i = 0; i < ni; i++)
    {
        mu[i] = (float) rand() / (float) RAND_MAX;
    }
    for (i = ni+nb+np; i < ns; i++)
    {
        mu[i] = (float) rand() / (float) RAND_MAX;
    }
    _normalizeMu(mu, ni, nb, np);
}

void InitUniformCentroids(Destin *d, uint l, uint ni, uint nb, uint np, uint ns)
{
    uint i, j;

    MALLOCV(d->uf_mu[l], float *, nb);
    MALLOCV(d->uf_sigma[l], float *, nb);
    MALLOCV(d->uf_absvar[l], float, ns);

    MALLOCV(d->uf_winCounts[l], uint, nb);
    MALLOCV(d->uf_winFreqs[l], float, nb);
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
        d->uf_winFreqs[l][i] = 1/(float) nb;
        d->uf_persistWinCounts[l][i] = 0;
        d->uf_persistWinCounts_detailed[l][i] = 0;
        d->uf_starv[l][i] = 1;

        _initUniformCentroidMu(d->uf_mu[l][i], ni, nb, np, ns);
        for (j=0; j < ns; j++)
        {
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
    uint i, j, ni, nb, np, ns;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ni = n->ni;
    np = n->np;
    ns = n->ns;

    // Layer l
    for (i = 0; i < nb; i++)
    {
        ArrayDeleteFloat(&d->uf_mu[l][i], ns, ni+idx);
        _normalizeMu(d->uf_mu[l][i], ni, nb-1, np);
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
    ArrayDeleteFloat(&d->uf_starv[l], nb, idx);

    ArrayDeleteFloat(&d->uf_winFreqs[l], nb, idx);
    _normalizeFloatArray(d->uf_winFreqs[l], nb - 1);

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
        uint childIndexes[n->nChildren];  // indexes of deleted centroids for all childs

        for (i = 0; i < n->nChildren; i++)
        {
            childIndexes[i] = i*nb + idx;
        }
        for (i = 0; i < n->nb; i++)
        {
            ArrayDeleteFloats(&d->uf_mu[l+1][i], ns, childIndexes, n->nChildren);
            ArrayDeleteFloats(&d->uf_sigma[l+1][i], ns, childIndexes, n->nChildren);
            ArrayDeleteFloats(&d->uf_avgDelta[l+1][i], ns, childIndexes, n->nChildren);
            ArrayDeleteFloats(&d->uf_avgSquaredDelta[l+1][i], ns, childIndexes, n->nChildren);
        }
        ArrayDeleteFloats(&d->uf_absvar[l+1], ns, childIndexes, n->nChildren);
        ArrayDeleteFloats(&d->uf_avgAbsDelta[l+1], ns, childIndexes, n->nChildren);

        for (j = 0; j < d->layerSize[l+1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l+1] + j];

            ArrayDeleteFloats(&n->delta, ns, childIndexes, n->nChildren);
            ArrayDeleteFloats(&n->observation, ns, childIndexes, n->nChildren);

            // decrease centroid dimensionality for each node from layer l+1
            UpdateNodeSizes(n, n->ni - n->nChildren, n->nb, n->np, n->nc);

            // pointers may change due to reallocation
            n->mu = d->uf_mu[l+1];
            n->starv = d->uf_starv[l+1];
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
            _normalizeMu(d->uf_mu[l-1][i], n->ni, n->nb, n->np-1);
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

            // pointers may change due to reallocation
            n->mu = d->uf_mu[l-1];
            n->starv = d->uf_starv[l-1];
        }
    }
}

void AddUniformCentroid(Destin *d, uint l)
{
    uint i, j, ni, nb, np, ns, idx;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ni = n->ni;
    np = n->np;
    ns = n->ns;
    idx = ni + nb;

    // Layer l
    float * newMu, * newSigma, * newAvgDelta, * newAvgSquaredDelta;

    MALLOCV(newMu, float, (ns+1));
    MALLOCV(newSigma, float, (ns+1));
    MALLOCV(newAvgDelta, float, (ns+1));
    MALLOCV(newAvgSquaredDelta, float, (ns+1));

    _initUniformCentroidMu(newMu, ni, nb+1, np, ns+1);
    for (j=0; j < ns+1; j++)
    {
        newSigma[j] = INIT_SIGMA;
        newAvgDelta[j] = 0;
        newAvgSquaredDelta[j] = 0;
    }

    for (i = 0; i < nb; i++)
    {
        ArrayInsertFloat(&d->uf_mu[l][i], ns, idx, 0);
        _normalizeMu(d->uf_mu[l][i], ni, nb+1, np);
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
    ArrayAppendFloat(&d->uf_starv[l], nb, 1);

    ArrayAppendFloat(&d->uf_winFreqs[l], nb, 1/(float) nb);
    _normalizeFloatArray(d->uf_winFreqs[l], nb+1);

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
        uint childIndexes[n->nChildren];  // indexes of added centroids for all childs
        float childValues[n->nChildren];  // initial values for all childs
        float childSigmas[n->nChildren];

        for (i = 0; i < n->nChildren; i++)
        {
            childIndexes[i] = (i+1)*nb;
            childValues[i] = 0;
            childSigmas[i] = INIT_SIGMA;
        }
        for (i = 0; i < n->nb; i++)
        {
            ArrayInsertFloats(&d->uf_mu[l+1][i], ns, childIndexes, childValues, n->nChildren);
            ArrayInsertFloats(&d->uf_sigma[l+1][i], ns, childIndexes, childSigmas, n->nChildren);
            ArrayInsertFloats(&d->uf_avgDelta[l+1][i], ns, childIndexes, childValues, n->nChildren);
            ArrayInsertFloats(&d->uf_avgSquaredDelta[l+1][i], ns, childIndexes, childValues, n->nChildren);
        }
        ArrayInsertFloats(&d->uf_absvar[l+1], ns, childIndexes, childValues, n->nChildren);
        ArrayInsertFloats(&d->uf_avgAbsDelta[l+1], ns, childIndexes, childValues, n->nChildren);

        for (j = 0; j < d->layerSize[l+1]; j++)
        {
            n =& d->nodes[d->layerNodeOffsets[l+1] + j];

            ArrayInsertFloats(&n->delta, ns, childIndexes, childValues, n->nChildren);
            ArrayInsertFloats(&n->observation, ns, childIndexes, childValues, n->nChildren);

            // increase centroid dimensionality for each node from layer l+1
            UpdateNodeSizes(n, n->ni + n->nChildren, n->nb, n->np, n->nc);

            // pointers may change due to reallocation
            n->mu = d->uf_mu[l+1];
            n->starv = d->uf_starv[l+1];
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
            _normalizeMu(d->uf_mu[l-1][i], n->ni, n->nb, n->np+1);
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

            // pointers may change due to reallocation
            n->mu = d->uf_mu[l-1];
            n->starv = d->uf_starv[l-1];
        }
    }

    // TODO: use proper method based on learning strategy
    InitUniformCentroidByAvgNearNeighbours(d, l, nb, 5);
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
    float sum = 0;                                   \
    for (c = 1; c <= count; c++) {                   \
        sum += array[offset + neighbours[c].index];  \
    }                                                \
    array[offset + idx] = sum/count;                 \
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
    // TODO: this is irrelevant if recursive mode is off
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
            for (j = 0; j < n->nChildren; j++)
            {
                UpdateAvgNearNeighbours(d->uf_mu[l+1][i], neighbours, nearest, j*nb, idx);
                UpdateAvgNearNeighbours(d->uf_sigma[l+1][i], neighbours, nearest, j*nb, idx);
            }
        }
    }

    // Layer l-1
    // TODO: this is irrelevant if recursive mode is off
    if (l > 0)
    {
        n = GetNodeFromDestin(d, l-1, 0, 0);
        uint offset = n->ni + n->nb;

        for (i = 0; i < n->nb; i++)
        {
            UpdateAvgNearNeighbours(d->uf_mu[l-1][i], neighbours, nearest, offset, idx);
            UpdateAvgNearNeighbours(d->uf_sigma[l-1][i], neighbours, nearest, offset, idx);
            _normalizeFloatArray(d->uf_mu[l-1][i] + offset, nb);
        }
    }
}


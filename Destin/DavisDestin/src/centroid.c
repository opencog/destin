#include <stdlib.h>

#include "macros.h"
#include "centroid.h"
#include "destin.h"
#include "array.h"

#define NEAREST_OF_DELETED_CENTROID     5
#define VICINITY_OF_ADDED_CENTROID      1

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

    // Recurrent mode off
    for (i = ni; i < ni+nb; i++)
    {
        mu[i] = 1/(float) nb;
    }
    // Recurrent mode off
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

void _distributeEvidenceOfDeletedCentroidToNeighbours(Destin *d, uint l, uint idx, uint nearest);

void DeleteUniformCentroid(Destin *d, uint l, uint idx)
{
    uint i, j, ni, nb, np, ns;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ni = n->ni;
    np = n->np;
    ns = n->ns;

    _distributeEvidenceOfDeletedCentroidToNeighbours(d, l, idx, NEAREST_OF_DELETED_CENTROID);

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

void _generateNewUniformCentroidMu(Destin *d, uint l, uint vicinity, float * newMu);

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

    _generateNewUniformCentroidMu(d, l, VICINITY_OF_ADDED_CENTROID, newMu);
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

}

// Private structure for q-sorting neighbouring centroids
typedef struct _Neighbour {
    uint index;      // index in centroid arrays (0..nb-1)
    float distance;  // distance from given centroid
    float weight;    // closer centroids have higher weights
} _Neighbour;

// Private comparator of neighbouring centroids by distance (in ascending order)
int _compareNeighboursByDistance(void * n1, void * n2)
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

// The method used to distribute values of beliefs associated with centroid that
// is going to be deleted. It helps when deleted centroid is very close to good
// learned centroids. This centroids from neighbourhood get additional weighted
// belief of deleted centroid where more weights have centroids that are closer.
void _distributeEvidenceOfDeletedCentroidToNeighbours(Destin *d, uint l, uint idx, uint nearest)
{
    uint i, j, k, nb, ns;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    nb = n->nb;
    ns = n->ns;

    if (nb < 2)
    {
        fprintf (stderr, "DistributeWeightsOfDeletedCentroid(): called when the layer %d has only single centroid!", l);
        return;
    }

    _Neighbour neighbours[nb];
    for (i = 0; i < nb; i++)
    {
        neighbours[i].index = i;
        neighbours[i].distance = _calcNeighboursDistanceEuc(d->uf_mu[l][i], d->uf_mu[l][idx], ns) + EPSILON;
        neighbours[i].weight = 0;
    }
    qsort((void *)neighbours, nb, sizeof(_Neighbour), (void *)&_compareNeighboursByDistance);
    nearest = _MIN(nearest, nb-1);

    float sumDistance = 0;
    for (i = 1; i <= nearest; i++)
    {
        sumDistance += neighbours[i].distance;
    }
    float sumWeights = 0;
    for (i = 1; i <= nearest; i++)
    {
        neighbours[i].weight = sumDistance / neighbours[i].distance;
        sumWeights += neighbours[i].weight;
    }
    for (i = 1; i <= nearest; i++)
    {
        neighbours[i].weight /= sumWeights;
    }

    // Layer l+1
    if (l+1 < d->nLayers)
    {
        n = GetNodeFromDestin(d, l+1, 0, 0);
        for (i = 0; i < n->nb; i++)
        {
            for (j = 0; j < n->nChildren; j++)
            {
                for (k = 1; k <= nearest; k++)
                {
                    d->uf_mu[l+1][i][nb*j + neighbours[k].index] += d->uf_mu[l+1][i][nb*j + idx] * neighbours[k].weight;
                    d->uf_sigma[l+1][i][nb*j + neighbours[k].index] += d->uf_sigma[l+1][i][nb*j + idx] * neighbours[k].weight;
                }
                // centroid evidence has been distributed
                d->uf_mu[l+1][i][nb*j + idx] = 0;
                d->uf_sigma[l+1][i][nb*j + idx] = 0;
            }
        }
    }

    // TODO: RECURRENT MODE: move evidence for l layer and l-1 layer
}

void _sampleNormalDistributedCentroid(float * mu, float * centerMu, float * centerSigma, uint ns)
{
    uint i;
    float u1, u2;

    for (i = 0; i < ns; i++)
    {
        u1 = rand() / (float) RAND_MAX;
        u2 = rand() / (float) RAND_MAX;

        mu[i] = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2) * centerSigma[i] + centerMu[i];
        mu[i] = (mu[i] > 1) ? 1 : mu[i];
        mu[i] = (mu[i] < 0) ? 0 : mu[i];
    }
}

void _generateNewUniformCentroidMu(Destin *d, uint l, uint vicinity, float * newMu)
{
    uint i, j, k, ni, nb, np, ns;
    float sigma;

    Node * n = GetNodeFromDestin(d, l, 0, 0);
    ni = n->ni;
    nb = n->nb;
    np = n->np;
    ns = n->ns;

    float weights[nb];
    uint picked[vicinity];
    float pickedWeights[vicinity];

    // calculate weights
    float sumWeight = 0;
    for (i = 0; i < nb; i++)
    {
        sigma = 0;
        for (j = 0; j < ns; j++)
        {
            sigma += d->uf_sigma[l][i][j];
        }
        weights[i] = sigma; // d->uf_winFreqs[l][i] * sigma;
        sumWeight += weights[i];
    }

    // pick centroids with probability proportional to weights
    float sumPickedWeights = 0;
    for (i = 0; i < vicinity; i++)
    {
        float pickWeight = 0;
        float rnd = sumWeight * rand() / (float) RAND_MAX;
        for (j = 0; (j < nb) && (rnd >= pickWeight); j++)
        {
            pickWeight += weights[j];
        }
        picked[i] = j - 1;
        pickedWeights[i] = rand() / (float) RAND_MAX;
        sumPickedWeights += pickedWeights[i];
    }

    float centerMu[ns];
    float centerSigma[ns];
    for (i = 0; i < ns; i++)
    {
        centerMu[i] = 0;
        centerSigma[i] = INIT_SIGMA;
        for (j = 0; j < vicinity; j++)
        {
            centerMu[i] += d->uf_mu[l][picked[j]][i] * pickedWeights[j] / sumPickedWeights;
        }
    }
    _sampleNormalDistributedCentroid(newMu, centerMu, centerSigma, ns);

    // TODO: in recurrent mode take care about position nb+1
    _normalizeMu(newMu, ni, nb+1, np);
}


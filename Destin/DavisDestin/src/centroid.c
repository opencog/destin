#include "macros.h"
#include "centroid.h"
#include "destin.h"

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

// 2013.6.6
// CZT
// killCentroid
void killCentroid(Destin * d, uint *nci, uint nl, uint *nb, uint nc, float beta, float lambda, float gamma, float *temp, float starvCoeff, uint nMovements, bool isUniform,
                  int extRatio, int currLayer, int kill_ind, float ** sharedCen, float ** starv, float ** sigma,
                  long **persistWinCounts, long ** persistWinCounts_detailed, float ** absvar)
{
    uint i, l, maxNb, maxNs, j;

    initializeDestinParameters(nb, isUniform, nci, extRatio, nl, nMovements, d, nc, temp);
        // uf_starv
        // uf_persistWinCounts
    if(isUniform){
        for(l=0; l<d->nLayers; ++l)
        {
            if(l == currLayer)
            {
                for(i=0; i<kill_ind; ++i)
                {
                    d->uf_starv[l][i] = starv[l][i];
                    d->uf_persistWinCounts[l][i] = persistWinCounts[l][i];
                    d->uf_persistWinCounts_detailed[l][i] = persistWinCounts_detailed[l][i];
                }
                for(i=kill_ind+1; i<d->nb[l]+1; ++i)
                {
                    d->uf_starv[l][i-1] = starv[l][i];
                    d->uf_persistWinCounts[l][i-1] = persistWinCounts[l][i];
                    d->uf_persistWinCounts_detailed[l][i-1] = persistWinCounts_detailed[l][i];
                }
            }
            else
            {
                for(i=0; i<d->nb[l]; ++i)
                {
                    d->uf_starv[l][i] = starv[l][i];
                    d->uf_persistWinCounts[l][i] = persistWinCounts[l][i];
                    d->uf_persistWinCounts_detailed[l][i] = persistWinCounts_detailed[l][i];
                }
            }
        }
    }

    // keep track of the max num of beliefs and states.  we need this information
    // to correctly call kernels later
    maxNb = 0;
    maxNs = 0;

    uint n = 0;

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
            //
            MALLOC(d->uf_absvar[l], float, ns*nb[l]);
            //
            MALLOC(d->uf_avgSquaredDelta[l], float, ns*nb[l]);
            MALLOC(d->uf_avgAbsDelta[l], float, ns*nb[l]);

            // 2013.6.4
            // Keep the shareCentroids, uf_avgDelta, uf_sigma
            if(l == currLayer)
            {
                int tempI, tempJ;
                int tempRange = l==0?ni*(d->extRatio-1):0;
                for(tempI=0; tempI<kill_ind; ++tempI)
                {
                    for(tempJ=0; tempJ<ns-d->nb[l]-np-tempRange+kill_ind; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns+1)+tempJ];
                    }
                    for(tempJ=ns-d->nb[l]-np-tempRange+kill_ind+1; tempJ<ns+1; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ-1] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ-1] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ-1] = absvar[l][tempI*(ns+1)+tempJ];
                    }
                }
                for(tempI=kill_ind+1; tempI<d->nb[l]+1; ++tempI)
                {
                    for(tempJ=0; tempJ<ns-d->nb[l]-np-tempRange+kill_ind; ++tempJ)
                    {
                        sharedCentroids[(tempI-1)*ns+tempJ] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][(tempI-1)*ns+tempJ] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][(tempI-1)*ns+tempJ] = absvar[l][tempI*(ns+1)+tempJ];
                    }
                    for(tempJ=ns-d->nb[l]-np-tempRange+kill_ind+1; tempJ<ns+1; ++tempJ)
                    {
                        sharedCentroids[(tempI-1)*ns+tempJ-1] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][(tempI-1)*ns+tempJ-1] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][(tempI-1)*ns+tempJ-1] = absvar[l][tempI*(ns+1)+tempJ];
                    }
                }

            }
            // currLayer-1
            else if(l==currLayer-1)
            {
                int tempI, tempJ;
                int tempRange = l==0?ni*(d->extRatio-1):0;
                for(tempI=0; tempI<d->nb[l]; ++tempI)
                {
                    for(tempJ=0; tempJ<ns-np-tempRange+kill_ind; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns+1)+tempJ];
                    }
                    for(tempJ=ns-np-tempRange+kill_ind+1; tempJ<ns+1; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ-1] = sharedCen[l][tempI*(ns+1)+tempJ];
                        d->uf_sigma[l][tempI*ns+tempJ-1] = sigma[l][tempI*(ns+1)+tempJ];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ-1] = absvar[l][tempI*(ns+1)+tempJ];
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
                        for(tempK=0; tempK<kill_ind; ++tempK)
                        {
                            sharedCentroids[tempI*ns+tempJ*d->nb[l-1]+tempK] = sharedCen[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                            d->uf_sigma[l][tempI*ns+tempJ*d->nb[l-1]+tempK] = sigma[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                            //
                            d->uf_absvar[l][tempI*ns+tempJ*d->nb[l-1]+tempK] = absvar[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                        }
                        for(tempK=kill_ind+1; tempK<nb[l-1]+1; ++tempK)
                        {
                            sharedCentroids[tempI*ns+tempJ*d->nb[l-1]+tempK-1] = sharedCen[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                            d->uf_sigma[l][tempI*ns+tempJ*d->nb[l-1]+tempK-1] = sigma[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                            //
                            d->uf_absvar[l][tempI*ns+tempJ*d->nb[l-1]+tempK-1] = absvar[l][tempI*(ns+childs)+tempJ*(d->nb[l-1]+1)+tempK];
                        }
                    }
                    for(tempJ=childs*d->nb[l-1]; tempJ<ns; ++tempJ)
                    {
                        sharedCentroids[tempI*ns+tempJ] = sharedCen[l][tempI*(ns+childs)+tempJ+childs];
                        d->uf_sigma[l][tempI*ns+tempJ] = sigma[l][tempI*(ns+childs)+tempJ+childs];
                        //
                        d->uf_absvar[l][tempI*ns+tempJ] = absvar[l][tempI*(ns+childs)+tempJ+childs];
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